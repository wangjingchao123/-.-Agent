/* 银龄安动 - 训练模块
 * 训练流程控制 + 摄像头/MediaPipe姿态识别 + 姿势语音纠正 + 体感询问动态调难度
 * 无摄像头/无MediaPipe时自动降级为"语音引导模式"，流程照常推进
 */
(function () {
  'use strict';
  window.SA = window.SA || {};

  var REP_PACE = 3;          // 计数型动作节奏：每3秒1次
  var REST_SECONDS = 5;      // 动作间休息秒数
  var CORRECTION_COOLDOWN = 8000;  // 同一提醒冷却毫秒
  var FEELING_EVERY = 2;     // 每完成N个动作询问一次体感

  var plan = [], idx = 0, ex = null;
  var target = 0, progress = 0, difficulty = 1;
  var tickId = null, restId = null, frameId = null;
  var pose = null, poseReady = false, stream = null;
  var lastCorrection = {};   // {提醒文本: 上次时间戳}
  var stats = { completed: 0, skipped: 0, feelings: [], startAt: 0, seconds: 0 };
  var feelingCb = null;

  /* ---------- 示范→等待确认→正式训练 三阶段 ---------- */
  var DEMO_STEP_MS = 3000;   // 示范视频节拍：每步3秒，训练引导同节拍对齐
  var phase = 'idle';        // demo(示范播放) | waiting(等待确认) | training(正式训练) | idle
  var demoTimer = null, demoStepIdx = 0, demoStepsTotal = 0;
  var okStreak = 0;          // OK手势连续命中帧数（连续2帧才触发，防误识别）
  var waitListenActive = false;

  function $(id) { return SA.$ && SA.$(id); }
  function isHeadless() { return !!(navigator && navigator.webdriver); }
  function speak(text) {
    return (SA.voice && SA.voice.enabled) ? SA.voice.speak(text) : Promise.resolve();
  }

  /* ---------- 可测纯函数 ---------- */
  /* 按难度系数计算目标值：次数向上取整，时长取整 */
  function calcTarget(exercise, diff) {
    if (!exercise || !exercise.metric) return 0;
    var v = exercise.metric.value * (diff || 1);
    return exercise.metric.type === 'count' ? Math.ceil(v) : Math.round(v);
  }

  /* 根据体感反馈调整难度：good不变 / tired降至0.7倍 / dizzy停止 */
  function applyFeeling(diff, feeling) {
    /* 保留两位小数，避免浮点误差随多次反馈累积 */
    if (feeling === 'tired') return { difficulty: Math.max(0.4, Math.round(diff * 0.7 * 100) / 100), stop: false };
    if (feeling === 'dizzy') return { difficulty: diff, stop: true };
    return { difficulty: diff, stop: false };
  }

  /* ---------- 流程控制 ---------- */
  function start() {
    plan = (SA.state && SA.state.plan) || [];
    if (!plan.length) {
      if (SA.voice) SA.voice.speak('还没给您准备好方案呢，咱们先聊几句身体情况，马上就好。');
      SA.go('interview');
      return;
    }
    idx = 0; difficulty = 1;
    stats = { completed: 0, skipped: 0, feelings: [], startAt: Date.now(), seconds: 0 };
    lastCorrection = {};
    SA.state.trainStats = stats;
    SA.go('train');
    initCamera();
    startExercise(0);
  }

  function stop(reason) {
    clearTimers();
    resetDemoAndWait();
    stats.seconds = Math.round((Date.now() - stats.startAt) / 1000);
    if (reason) stats.feelings.push(reason);
    stopCamera();
    var p = (SA.voice && SA.voice.enabled)
      ? SA.voice.speak(reason === 'dizzy'
        ? '觉得不舒服就停下来，安全第一。您先歇会儿，今天的锻炼就到这儿。'
        : '今天的锻炼就到这儿啦，您辛苦了，真棒！')
      : Promise.resolve();
    p.then(function () { finishToSummary(); });
  }

  /* ---------- 紧急暂停（身体不舒服大按钮） ---------- */
  /* 立即停止摄像头识别与训练，语音安抚，大字提示就医，不跳总结页 */
  function emergencyStop() {
    clearTimers();
    stopCamera();
    hideFeeling();
    resetDemoAndWait();
    if (SA.voice) { SA.voice.stopSpeak(); SA.voice.abortListen(); }
    stats.seconds = Math.round((Date.now() - stats.startAt) / 1000);
    stats.feelings.push('emergency');
    SA.state.trainStats = stats;
    var ov = $('emergency-overlay');
    if (ov) ov.hidden = false;
    if (SA.voice && SA.voice.enabled) {
      SA.voice.speak('别担心，训练已经停下来了。您先坐好，慢慢喘口气，休息一会儿。要是觉得心慌、胸口疼、或者头晕得厉害，请马上告诉家里人，或者拨打一二零。身体最要紧，咱们不着急。');
    }
  }

  function closeEmergency() {
    var ov = $('emergency-overlay');
    if (ov) ov.hidden = true;
    SA.go('home');
  }

  function finishToSummary() {
    stats.seconds = Math.round((Date.now() - stats.startAt) / 1000);
    SA.state.trainStats = stats;
    SA.go('summary');
    renderSummary();
  }

  function startExercise(i) {
    idx = i;
    ex = plan[i];
    if (!ex) { finishToSummary(); return; }
    target = calcTarget(ex, difficulty);
    progress = 0;
    lastCorrection = {};
    noPersonActive = false;
    phase = 'demo';

    var nameEl = $('train-exercise-name'), descEl = $('train-exercise-desc'),
        tipEl = $('train-tip'), progEl = $('train-progress'),
        cntEl = $('train-count-num'), timeEl = $('train-time-num');
    if (nameEl) nameEl.textContent = ex.name;
    if (descEl) descEl.textContent = ex.desc;
    if (tipEl) tipEl.textContent = ex.tip || '跟着语音提示慢慢来';
    if (progEl) progEl.textContent = (i + 1) + ' / ' + plan.length;
    if (cntEl) cntEl.textContent = '0';
    if (timeEl) timeEl.textContent = '0';
    hideFeeling();
    hideWaitConfirm();

    /* 先播放示范视频（步骤动画+语音讲解），看完关闭后进入等待确认，不自动开练 */
    showDemo();
    speak(buildIntro(i));
  }

  /* 温和开场白：太极招式播报"第几式保持多久"，常规动作播报序号 */
  function buildIntro(i) {
    var s;
    if (ex.category === '24式太极') {
      s = '好，咱们来做' + ex.name + '。';
      if (ex.metric && ex.metric.type === 'seconds') s += '这个招式咱们保持' + target + '秒。';
      else s += '这个招式咱们做' + target + '次。';
    } else {
      s = '好，咱们来做第' + (i + 1) + '个动作，叫' + ex.name + '。';
    }
    s += '先看一遍示范，不着急。';
    return s;
  }

  /* 计数型：节拍对齐示范视频（REP_PACE=示范每步时长），语音循环播报步骤要领 */
  function startRepCount() {
    clearTick();
    var up = true;
    var steps = (ex && ex.steps) || [];
    var stepPos = 0;
    tickId = setInterval(function () {
      if (up) {
        progress++;
        var cntEl = $('train-count-num');
        if (cntEl) cntEl.textContent = String(progress);
        if (progress >= target) {
          clearTick();
          speak('好，这个动作完成啦');
          finishExercise();
          return;
        }
      }
      /* 每拍播报一个步骤要领，循环流动，不再机械蹦"起/落" */
      if (steps.length) {
        speak(steps[stepPos % steps.length]);
        stepPos++;
      } else {
        speak(up ? '慢慢起' : '慢慢落');
      }
      up = !up;
    }, REP_PACE * 1000);
  }

  /* 计时型：倒计时，每10秒报时；同时按示范节拍循环播报步骤要领 */
  function startCountdown() {
    clearTick();
    var steps = (ex && ex.steps) || [];
    var stepPos = 0;
    var beatSec = Math.max(1, Math.round(DEMO_STEP_MS / 1000));
    tickId = setInterval(function () {
      progress++;
      var timeEl = $('train-time-num');
      if (timeEl) timeEl.textContent = String(progress);
      var left = target - progress;
      var spoke = false;
      if (left > 0 && left % 10 === 0) {
        speak('还剩' + left + '秒，您坚持得真好');
        spoke = true;
      }
      /* 节拍对齐示范：每 beatSec 秒播报一个步骤要领 */
      if (!spoke && steps.length && progress % beatSec === 0) {
        speak(steps[stepPos % steps.length]);
        stepPos++;
      }
      if (progress >= target) { clearTick(); finishExercise(); }
    }, 1000);
  }

  function finishExercise() {
    stats.completed++;
    var p = (SA.voice && SA.voice.enabled) ? SA.voice.speak('这个动作做完啦，您真棒！咱们歇一会儿。') : Promise.resolve();
    p.then(function () {
      if (idx + 1 >= plan.length) { finishToSummary(); return; }
      if (stats.completed % FEELING_EVERY === 0) { askFeeling(); return; }
      restThenNext();
    });
  }

  function restThenNext() {
    clearRest();
    restId = setTimeout(function () { startExercise(idx + 1); }, REST_SECONDS * 1000);
  }

  /* ---------- 体感询问 ---------- */
  function askFeeling() {
    showFeeling();
    if (SA.voice && SA.voice.enabled) {
      SA.voice.speak('您现在感觉怎么样呀？觉得很好，就说继续；要是有点累，就说轻松点；要是头晕不舒服，就说停止。');
    }
    /* 语音听答（无头环境跳过） */
    if (!isHeadless() && SA.voice && SA.voice.enabled && SA.voice.isSTTSupported()) {
      SA.voice.listen().then(function (result) {
        if (SA.voice.matchAny(result, ['头晕', '不舒服', '难受', '停止'])) handleFeeling('dizzy');
        else if (SA.voice.matchAny(result, ['累', '轻松', '吃力', '喘'])) handleFeeling('tired');
        else if (SA.voice.matchAny(result, ['好', '继续', '可以', '没问题'])) handleFeeling('good');
      }).catch(function () { /* 保持按钮等待 */ });
    }
  }

  function handleFeeling(feeling) {
    stats.feelings.push(feeling);
    hideFeeling();
    if (SA.voice) SA.voice.abortListen();
    var r = applyFeeling(difficulty, feeling);
    difficulty = r.difficulty;
    if (r.stop) { stop('dizzy'); return; }
    if (feeling === 'tired' && SA.voice && SA.voice.enabled) {
      SA.voice.speak('好嘞，那咱们接下来轻松点，慢慢来，不着急。');
    }
    restThenNext();
  }

  function showFeeling() {
    var p = $('feeling-panel');
    if (p) p.hidden = false;
  }
  function hideFeeling() {
    var p = $('feeling-panel');
    if (p) p.hidden = true;
  }

  /* ---------- 示范视频弹窗 ---------- */
  function showDemo() {
    var ov = $('demo-overlay');
    if (!ov) { enterWaiting(); return; }   /* 无弹窗DOM（如精简测试页）直接进入等待 */
    var nameEl = $('demo-exercise-name'), descEl = $('demo-desc');
    if (nameEl) nameEl.textContent = ex.name;
    if (descEl) descEl.textContent = ex.desc || '';
    var steps = ex.steps || [];
    demoStepsTotal = steps.length;
    var dots = $('demo-step-dots');
    if (dots) {
      dots.innerHTML = '';
      steps.forEach(function (_, k) {
        var d = document.createElement('span');
        d.className = 'demo-dot';
        dots.appendChild(d);
      });
    }
    ov.hidden = false;
    startDemoAnim();
  }

  /* 示范动画：按 DEMO_STEP_MS 节拍循环播放步骤，同步语音讲解（节拍即训练引导节拍） */
  function startDemoAnim() {
    clearDemoTimer();
    var steps = (ex && ex.steps) || [];
    if (!steps.length) return;
    demoStepIdx = 0;
    renderDemoStep(0);
    speak(steps[0]);
    demoTimer = setInterval(function () {
      if (phase !== 'demo') { clearDemoTimer(); return; }
      demoStepIdx = (demoStepIdx + 1) % steps.length;
      renderDemoStep(demoStepIdx);
      speak(steps[demoStepIdx]);
    }, DEMO_STEP_MS);
  }

  function renderDemoStep(k) {
    var steps = (ex && ex.steps) || [];
    var textEl = $('demo-step-text');
    if (textEl) textEl.textContent = steps[k] || '';
    var dots = $('demo-step-dots');
    if (dots) {
      for (var j = 0; j < dots.children.length; j++) {
        dots.children[j].classList.toggle('on', j === k);
      }
    }
    var bar = $('demo-progress-bar');
    if (bar && demoStepsTotal) {
      bar.style.width = (((k + 1) / demoStepsTotal) * 100) + '%';
    }
  }

  function clearDemoTimer() {
    if (demoTimer) { clearInterval(demoTimer); demoTimer = null; }
  }
  function hideDemo() {
    var ov = $('demo-overlay');
    if (ov) ov.hidden = true;
  }

  /* 看完示范、关闭弹窗 → 进入等待确认状态（不自动开练） */
  function closeDemo() {
    if (phase !== 'demo') return;
    clearDemoTimer();
    hideDemo();
    if (SA.voice) SA.voice.stopSpeak();
    enterWaiting();
  }

  /* ---------- 等待确认开始 ---------- */
  function enterWaiting() {
    phase = 'waiting';
    okStreak = 0;
    showWaitConfirm();
    var tipEl = $('train-tip');
    if (tipEl) tipEl.textContent = '准备好了就开始，不着急';
    speak('示范看完啦。准备好了的话，请比一个 OK 手势，或者对我说"可以"，咱们就开始。')
      .then(function () {
        if (phase === 'waiting') startWaitListen();
      });
  }

  function showWaitConfirm() {
    var w = $('wait-confirm');
    if (w) w.hidden = false;
  }
  function hideWaitConfirm() {
    var w = $('wait-confirm');
    if (w) w.hidden = true;
  }

  /* 语音循环监听"可以"（单次识别自动续听，匹配温和词组，排除"不行/不要"） */
  function startWaitListen() {
    if (phase !== 'waiting' || isHeadless()) return;
    if (!SA.voice || !SA.voice.enabled || !SA.voice.isSTTSupported()) return;
    waitListenActive = true;
    waitListenLoop();
  }

  function waitListenLoop() {
    if (!waitListenActive || phase !== 'waiting') return;
    SA.voice.listen().then(function (result) {
      if (!waitListenActive || phase !== 'waiting') return;
      if (SA.voice.matchAny(result, ['不行', '不要', '等等', '停一停', '暂停'])) {
        setTimeout(waitListenLoop, 1000);
        return;
      }
      if (SA.voice.matchAny(result, ['可以', '开始', '好了', '预备'])) {
        beginTraining();
        return;
      }
      setTimeout(waitListenLoop, 300);
    }).catch(function () {
      if (!waitListenActive || phase !== 'waiting') return;
      setTimeout(waitListenLoop, 700);
    });
  }

  function stopWaitListen() {
    waitListenActive = false;
    if (SA.voice) SA.voice.abortListen();
  }

  /* ---------- 确认开始 → 正式训练 ---------- */
  function beginTraining() {
    if (phase !== 'waiting') return;
    phase = 'training';
    stopWaitListen();
    hideWaitConfirm();
    okStreak = 0;
    if (SA.voice) SA.voice.stopSpeak();
    var tipEl = $('train-tip');
    if (tipEl) tipEl.textContent = ex.tip || '跟着语音提示慢慢来';
    speak('好嘞，咱们开始！跟着我的声音，慢慢来。').then(function () {
      if (phase !== 'training') return;
      if (ex.metric && ex.metric.type === 'seconds') startCountdown();
      else startRepCount();
    });
  }

  /* 统一清理示范/等待状态（停止/紧急暂停/退出训练时调用） */
  function resetDemoAndWait() {
    phase = 'idle';
    clearDemoTimer();
    stopWaitListen();
    hideDemo();
    hideWaitConfirm();
    okStreak = 0;
  }

  /* ---------- 摄像头与姿态识别 ---------- */
  function initCamera() {
    var statusEl = $('camera-status');
    var video = $('camera-video');
    if (statusEl) statusEl.hidden = false;   /* 重置遮罩，降级提示需可见 */
    if (isHeadless() || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      if (statusEl) statusEl.textContent = '语音引导模式（未开启摄像头）';
      return;
    }
    /* 超时降级：摄像头2.5秒未就绪即提示语音引导，避免老人长时间等待黑屏 */
    var timedOut = false;
    var timeoutId = setTimeout(function () {
      timedOut = true;
      if (statusEl) statusEl.textContent = '摄像头未就绪，已进入语音引导模式';
    }, 2500);
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } })
      .then(function (s) {
        if (timedOut) { s.getTracks().forEach(function (t) { t.stop(); }); return; }
        clearTimeout(timeoutId);
        stream = s;
        if (video) { video.srcObject = s; return video.play(); }
      })
      .then(function () {
        if (timedOut) return;
        if (statusEl) statusEl.textContent = '摄像头已开启，正在识别姿势…';
        /* 隐藏全屏状态遮罩，让摄像头画面正常预览（骨架绘制即识别反馈） */
        if (statusEl) statusEl.hidden = true;
        initPose();
      })
      .catch(function () {
        clearTimeout(timeoutId);
        if (statusEl) statusEl.textContent = '未检测到摄像头，已进入语音引导模式';
      });
  }

  function initPose() {
    if (typeof window.Pose !== 'function') {
      /* MediaPipe 未加载（无网络），仅显示摄像头画面 */
      return;
    }
    try {
      pose = new window.Pose({
        locateFile: function (f) { return 'https://cdn.jsdelivr.net/npm/@mediapipe/pose/' + f; }
      });
      pose.setOptions({
        modelComplexity: 0,
        smoothLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });
      pose.onResults(onPoseResults);
      poseReady = true;
      loopFrames();
    } catch (e) {
      poseReady = false;
    }
  }

  function loopFrames() {
    var video = $('camera-video');
    if (!poseReady || !pose || !video || video.readyState < 2) {
      frameId = requestAnimationFrame(loopFrames);
      return;
    }
    pose.send({ image: video }).catch(function () { /* 帧发送失败静默重试 */ });
    frameId = requestAnimationFrame(loopFrames);
  }

  /* 识别不到人：只给文字提示（节流6秒），不暂停、不弹紧急窗、不语音打扰 */
  var noPersonLast = 0, noPersonActive = false;
  function onPoseResults(results) {
    drawSkeleton(results);
    /* 等待确认阶段：只识别 OK 手势，不计数、不判定动作、不播报进度 */
    if (phase === 'waiting') {
      if (results.poseLandmarks && SA.pose && SA.pose.detectOKGesture &&
          SA.pose.detectOKGesture(results.poseLandmarks)) {
        okStreak++;
        if (okStreak >= 2) { okStreak = 0; beginTraining(); }
      } else {
        okStreak = 0;
      }
      return;
    }
    /* 仅正式训练阶段进行动作判定与纠错提醒（容错逻辑不变） */
    if (phase !== 'training' || !ex) return;
    if (!results.poseLandmarks) {
      var now = Date.now();
      if (now - noPersonLast > 6000) {
        noPersonLast = now;
        noPersonActive = true;
        var tipEl = $('train-tip');
        if (tipEl) tipEl.textContent = '没看到您呢，请站到摄像头前面，咱们接着练。';
      }
      return;
    }
    /* 重新识别到人：恢复当前动作提示文字 */
    if (noPersonActive) {
      noPersonActive = false;
      var tipEl2 = $('train-tip');
      if (tipEl2) tipEl2.textContent = ex.tip || '跟着语音提示慢慢来';
    }
    var r = SA.pose.evaluatePosture(results.poseLandmarks, ex);
    if (r.ok) return;
    var now2 = Date.now();
    r.messages.forEach(function (msg) {
      if (now2 - (lastCorrection[msg] || 0) < CORRECTION_COOLDOWN) return;
      lastCorrection[msg] = now2;
      var tipEl3 = $('train-tip');
      if (tipEl3) tipEl3.textContent = msg;
      speak(msg);
    });
  }

  /* 简化骨架绘制：连接关键骨骼线 + 关键点圆圈 */
  var BONES = [[11,12],[11,23],[12,24],[23,24],[11,13],[13,15],[12,14],[14,16],[23,25],[25,27],[24,26],[26,28]];
  function drawSkeleton(results) {
    var canvas = $('pose-canvas'), video = $('camera-video');
    if (!canvas || !video) return;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    var lm = results.poseLandmarks;
    if (!lm) return;
    ctx.lineWidth = 4;
    ctx.strokeStyle = 'rgba(80,220,140,0.9)';
    BONES.forEach(function (b) {
      var p1 = lm[b[0]], p2 = lm[b[1]];
      if (p1 && p2 && (p1.visibility || 1) > 0.5 && (p2.visibility || 1) > 0.5) {
        ctx.beginPath();
        ctx.moveTo(p1.x * canvas.width, p1.y * canvas.height);
        ctx.lineTo(p2.x * canvas.width, p2.y * canvas.height);
        ctx.stroke();
      }
    });
    ctx.fillStyle = 'rgba(255,215,0,0.95)';
    lm.forEach(function (p) {
      if ((p.visibility || 1) > 0.5) {
        ctx.beginPath();
        ctx.arc(p.x * canvas.width, p.y * canvas.height, 5, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  }

  function stopCamera() {
    poseReady = false;
    if (frameId) cancelAnimationFrame(frameId);
    frameId = null;
    if (stream) { stream.getTracks().forEach(function (t) { t.stop(); }); stream = null; }
    var video = $('camera-video');
    if (video) video.srcObject = null;
  }

  function clearTimers() { clearTick(); clearRest(); }
  function clearTick() { if (tickId) { clearInterval(tickId); tickId = null; } }
  function clearRest() { if (restId) { clearTimeout(restId); restId = null; } }

  /* ---------- 总结页渲染 ---------- */
  function renderSummary() {
    var box = $('summary-stats');
    if (!box) return;
    var s = SA.state.trainStats || stats;
    var mins = Math.max(1, Math.round(s.seconds / 60));
    box.innerHTML = '';
    [
      { v: s.completed + ' / ' + (plan.length || (SA.state.plan || []).length), l: '完成动作' },
      { v: mins + ' 分钟', l: '训练时长' },
      { v: s.feelings.indexOf('emergency') >= 0 ? '紧急暂停' : (s.feelings.indexOf('dizzy') >= 0 ? '因不适停止' : (s.feelings.length ? '已动态调整' : '状态良好')), l: '体感反馈' }
    ].forEach(function (it) {
      var card = document.createElement('div');
      card.className = 'stat-card';
      var v = document.createElement('div'); v.className = 'v'; v.textContent = it.v;
      var l = document.createElement('div'); l.className = 'l'; l.textContent = it.l;
      card.appendChild(v); card.appendChild(l);
      box.appendChild(card);
    });
    if (SA.voice && SA.voice.enabled) {
      SA.voice.speak('今天的锻炼完成啦！一共做了' + s.completed + '个动作，差不多' + mins + '分钟。您坚持锻炼，身体一定会越来越好的！');
    }
  }

  /* ---------- 暴露与绑定 ---------- */
  SA.train = {
    start: start,
    stop: stop,
    emergencyStop: emergencyStop,
    calcTarget: calcTarget,
    applyFeeling: applyFeeling,
    renderSummary: renderSummary,
    getDifficulty: function () { return difficulty; },
    closeDemo: closeDemo,
    beginTraining: beginTraining,
    getPhase: function () { return phase; }
  };
  SA.startTraining = start;

  document.addEventListener('DOMContentLoaded', function () {
    if (!SA.$) return;
    /* 体感面板按钮 */
    document.querySelectorAll('[data-feeling]').forEach(function (b) {
      b.addEventListener('click', function () { handleFeeling(b.getAttribute('data-feeling')); });
    });
    /* 退出训练按钮：温和确认 */
    var backBtn = document.querySelector('#screen-train [data-back]');
    if (backBtn) {
      backBtn.addEventListener('click', function () {
        clearTimers();
        resetDemoAndWait();
        stopCamera();
      });
    }
    /* 示范弹窗关闭 → 等待确认 */
    var demoCloseBtn = SA.$('btn-demo-close');
    if (demoCloseBtn) demoCloseBtn.addEventListener('click', closeDemo);
    /* 等待确认页"直接开始"兜底按钮 */
    var waitStartBtn = SA.$('btn-wait-start');
    if (waitStartBtn) waitStartBtn.addEventListener('click', beginTraining);
    /* 紧急暂停大按钮：立即停止一切，安抚+就医提示 */
    var emgBtn = SA.$('btn-emergency');
    if (emgBtn) emgBtn.addEventListener('click', emergencyStop);
    var emgHome = SA.$('btn-emergency-home');
    if (emgHome) emgHome.addEventListener('click', closeEmergency);
  });
})();