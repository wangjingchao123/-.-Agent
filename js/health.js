/* 银龄安动 - 健康问询模块
 * 语音对话采集：年龄段、慢性病、疼痛、跌倒史、站立能力
 * 采集结果写入 SA.state.health 健康标签，供方案生成使用
 * 交互设计：每题自动语音播报 -> 自动听答 -> 方言候选容错匹配 -> 按钮兜底
 */
(function () {
  'use strict';
  window.SA = window.SA || {};

  /* ---------- 问询题目定义 ---------- */
  var questions = [
    {
      id: 'age',
      type: 'single',
      question: '您好！请问您的年龄段是？',
      spoken: '爷爷、奶奶，您好呀！欢迎来到银龄安动。咱们先聊几句您的身体状况，好给您安排合适的锻炼。请问您今年多大岁数啦？是六十到六十九，七十到七十九，还是八十以上呢？',
      options: [
        { value: '60s', label: '60 - 69 岁', keywords: ['六十', '六十一', '六十二', '六十三', '六十四', '六十五', '六十六', '六十七', '六十八', '六十九'], tag: { ageGroup: '60s' } },
        { value: '70s', label: '70 - 79 岁', keywords: ['七十', '七十一', '七十二', '七十三', '七十四', '七十五', '七十六', '七十七', '七十八', '七十九'], tag: { ageGroup: '70s' } },
        { value: '80s', label: '80 岁以上', keywords: ['八十', '八十一', '八十二', '八十五', '九十', '九十五', '一百'], tag: { ageGroup: '80s' } }
      ]
    },
    {
      id: 'diseases',
      type: 'multi',
      question: '您有以下慢性病吗？（可多选）',
      spoken: '请问您有没有这些慢性病呀？比如高血压、膝关节炎、骨质疏松、糖尿病。要是都没有，您就说一声没有。',
      options: [
        { value: 'hypertension', label: '高血压', keywords: ['高血压', '血压高', '血压'], tag: { hypertension: true } },
        { value: 'kneeArthritis', label: '膝关节炎', keywords: ['膝关节炎', '膝盖关节炎', '关节炎', '膝盖退化', '关节'], tag: { kneeArthritis: true } },
        { value: 'osteoporosis', label: '骨质疏松', keywords: ['骨质疏松', '骨头疏松', '骨疏松', '疏松'], tag: { osteoporosis: true } },
        { value: 'diabetes', label: '糖尿病', keywords: ['糖尿病', '血糖高', '血糖'], tag: { diabetes: true } },
        { value: 'none', label: '都没有', keywords: ['没有', '都没', '无', '没什么', '没有慢性病'], tag: {} }
      ]
    },
    {
      id: 'pain',
      type: 'single',
      question: '最近您哪里经常疼痛吗？',
      spoken: '最近您身上有没有哪里经常疼呀？比如膝盖疼、腰背疼，还是没什么疼痛？',
      options: [
        { value: 'kneePain', label: '经常膝盖疼', keywords: ['膝盖疼', '膝盖痛', '膝关节疼', '膝关节痛', '腿疼', '腿痛', '膝盖'], tag: { kneePain: true } },
        { value: 'backPain', label: '腰背疼', keywords: ['腰疼', '腰痛', '背疼', '背痛', '腰背'], tag: { backPain: true } },
        { value: 'noPain', label: '没有明显疼痛', keywords: ['不疼', '不痛', '没有', '无明显', '还好', '没什么'], tag: {} }
      ]
    },
    {
      id: 'fall',
      type: 'single',
      question: '最近一年您跌倒过吗？',
      spoken: '最近这一年，您有没有跌倒过呀？摔过跤也算，您慢慢回忆一下。',
      options: [
        { value: 'yes', label: '跌倒过', keywords: ['跌倒', '摔倒', '摔过', '跌过', '摔了一跤', '有过', '跌了一跤'], tag: { fallHistory: true } },
        { value: 'no', label: '没有跌倒', keywords: ['没有', '没跌', '没摔', '没跌倒', '没有过'], tag: { fallHistory: false } }
      ]
    },
    {
      id: 'standing',
      type: 'single',
      question: '您平时站立的情况是？',
      spoken: '您平时站立的情况怎么样呀？能自己站稳十分钟，还是得扶着点东西，或者基本坐轮椅、卧床呢？',
      options: [
        { value: 'good', label: '能独立站稳 10 分钟', keywords: ['独立', '能站', '站稳', '自己站', '没问题', '可以站'], tag: { standing: 'good' } },
        { value: 'assist', label: '需要扶着东西', keywords: ['扶', '扶着', '需要扶', '扶东西', '拐杖', '拐棍', '助行', ' walker'], tag: { standing: 'assist' } },
        { value: 'poor', label: '基本坐轮椅或卧床', keywords: ['轮椅', '卧床', '躺着', '不能站', '坐轮椅', '起不来'], tag: { standing: 'poor' } }
      ]
    }
  ];

  /* ---------- 状态 ---------- */
  var current = 0;          // 当前题目下标
  var session = 0;          // 会话代次，防止跨题串扰的识别结果
  var multiSelected = {};   // 多选题已选 { value: true }
  var listening = false;

  function isHeadless() { return !!(navigator && navigator.webdriver); }

  /* ---------- 流程控制 ---------- */
  function start() {
    current = 0;
    session++;
    multiSelected = {};
    listening = false;
    SA.state.health = { notes: [] };
    SA.go('interview');
    ask(0);
  }

  function ask(i) {
    current = i;
    session++;               // 新题目开启新会话，旧识别结果作废
    multiSelected = {};
    listening = false;
    resetMicBtn();
    var q = questions[i];
    SA.$('interview-progress').textContent = '第 ' + (i + 1) + ' / ' + questions.length + ' 题';
    SA.$('interview-question').textContent = q.question;
    SA.$('interview-hint').textContent = q.type === 'multi'
      ? '可以说出病名，可多选；或点按下方按钮'
      : '您可以直接说出答案，或点按下方按钮';
    SA.$('interview-heard').hidden = true;
    renderOptions(q);
    var p = (SA.voice && SA.voice.enabled) ? SA.voice.speak(q.spoken || q.question) : Promise.resolve();
    p.then(function () {
      if (!isHeadless() && SA.voice && SA.voice.enabled && SA.voice.isSTTSupported()) {
        autoListen();
      }
    });
  }

  /* ---------- 选项渲染 ---------- */
  function renderOptions(q) {
    var box = SA.$('interview-answers');
    box.innerHTML = '';
    box.style.outline = '';
    q.options.forEach(function (opt) {
      var btn = document.createElement('button');
      btn.className = 'btn btn-primary answer-btn';
      btn.dataset.value = opt.value;
      btn.textContent = opt.label;
      btn.addEventListener('click', function () {
        if (q.type === 'multi') toggleMulti(q, opt);
        else selectSingle(q, opt, opt.label);
      });
      box.appendChild(btn);
    });
    if (q.type === 'multi') {
      var confirm = document.createElement('button');
      confirm.className = 'btn btn-warn';
      confirm.id = 'btn-multi-confirm';
      confirm.textContent = '✓ 确认选择';
      confirm.addEventListener('click', function () { confirmMulti(q); });
      box.appendChild(confirm);
    }
  }

  function updateMultiUI(q) {
    var btns = SA.$('interview-answers').querySelectorAll('.answer-btn');
    btns.forEach(function (b) {
      var on = !!multiSelected[b.dataset.value];
      b.classList.toggle('btn-primary', !on);
      b.classList.toggle('btn-ok', on);
      var base = labelOf(q, b.dataset.value);
      b.textContent = (on ? '✓ ' : '') + base;
    });
  }

  function labelOf(q, value) {
    for (var i = 0; i < q.options.length; i++) {
      if (q.options[i].value === value) return q.options[i].label;
    }
    return value;
  }

  /* ---------- 选择处理 ---------- */
  function toggleMulti(q, opt) {
    if (opt.value === 'none') {
      multiSelected = { none: true };
      updateMultiUI(q);
      setTimeout(function () { confirmMulti(q); }, 500);
      return;
    }
    delete multiSelected.none;
    if (multiSelected[opt.value]) delete multiSelected[opt.value];
    else multiSelected[opt.value] = true;
    updateMultiUI(q);
  }

  function confirmMulti(q) {
    var chosen = [];
    q.options.forEach(function (opt) {
      if (multiSelected[opt.value] && opt.value !== 'none') {
        applyTag(opt.tag);
        chosen.push(opt.label);
      }
    });
    var heard = chosen.length ? chosen.join('、') : '都没有';
    showHeard(heard);
    ackAndNext('哎，好嘞，我记下了。');
  }

  function selectSingle(q, opt, heard) {
    applyTag(opt.tag);
    showHeard(heard);
    ackAndNext('哎，好嘞，我记下了。');
  }

  function applyTag(tag) {
    if (!tag) return;
    Object.keys(tag).forEach(function (k) { SA.state.health[k] = tag[k]; });
  }

  function showHeard(text) {
    SA.$('heard-text').textContent = text;
    SA.$('interview-heard').hidden = false;
    SA.state.health.notes.push(text);
  }

  function ackAndNext(ackText) {
    if (SA.voice) { SA.voice.stopSpeak(); SA.voice.abortListen(); }
    stopListenUI();
    var p = (SA.voice && SA.voice.enabled) ? SA.voice.speak(ackText) : Promise.resolve();
    p.then(function () {
      if (current + 1 < questions.length) ask(current + 1);
      else finishInterview();
    });
  }

  function finishInterview() {
    if (SA.planner && SA.planner.generatePlan) {
      SA.state.plan = SA.planner.generatePlan(SA.state.health);
    }
    var p = (SA.voice && SA.voice.enabled)
      ? SA.voice.speak('好嘞，咱们聊完啦。我这就给您安排合适的锻炼动作，稍等一下下。')
      : Promise.resolve();
    p.then(function () { SA.go('plan'); });
  }

  /* ---------- 语音识别 ---------- */
  function autoListen() {
    if (listening) return;
    startListen();
  }

  function startListen() {
    if (!SA.voice || !SA.voice.isSTTSupported() || !SA.voice.enabled) return;
    var mySession = session;
    listening = true;
    var micBtn = SA.$('btn-mic');
    if (micBtn) { micBtn.classList.add('listening'); micBtn.textContent = '🎤 正在听…'; }
    SA.voice.listen().then(function (result) {
      listening = false;
      resetMicBtn();
      if (mySession !== session) return;   // 已切到下一题，丢弃过期结果
      handleVoiceResult(result);
    }).catch(function (err) {
      listening = false;
      resetMicBtn();
      if (mySession !== session) return;
      if (err && err.message && err.message.indexOf('没有听清') !== -1) {
        setTimeout(autoListen, 600);       // 静默重试
      } else if (err && err.message) {
        SA.voice.speak(err.message);
      }
    });
  }

  function stopListenUI() {
    listening = false;
    resetMicBtn();
  }

  function resetMicBtn() {
    var micBtn = SA.$('btn-mic');
    if (micBtn) { micBtn.classList.remove('listening'); micBtn.textContent = '🎤 说话'; }
  }

  function hasHeardText(result) {
    if (!result) return false;
    if (result.text) return true;
    return (result.alternatives || []).some(function (a) { return !!a.text; });
  }

  function handleVoiceResult(result) {
    var q = questions[current];
    if (!hasHeardText(result)) {
      SA.voice.speak('哎呀，没太听清。您再说一遍，或者点下面的按钮也行。').then(function () { autoListen(); });
      return;
    }
    if (q.type === 'single') {
      var matched = matchOption(q, result);
      if (matched) {
        selectSingle(q, matched, result.text);
      } else {
        SA.voice.speak('没听清您的回答，您再说一遍，或者点下面的按钮。').then(function () { autoListen(); });
      }
    } else {
      handleMultiVoice(q, result);
    }
  }

  /* 否定类选项优先检测：避免"没有高血压"先命中"高血压"而误判为患病 */
  function isNegative(opt) {
    return opt.value === 'none' || opt.value === 'noPain' || opt.value === 'no';
  }

  function matchOption(q, result) {
    var ordered = q.options.slice().sort(function (a, b) {
      return (isNegative(a) ? 0 : 1) - (isNegative(b) ? 0 : 1);
    });
    for (var i = 0; i < ordered.length; i++) {
      if (SA.voice.matchAny(result, ordered[i].keywords)) return ordered[i];
    }
    return null;
  }

  function handleMultiVoice(q, result) {
    var noneOpt = null;
    q.options.forEach(function (o) { if (o.value === 'none') noneOpt = o; });
    if (noneOpt && SA.voice.matchAny(result, noneOpt.keywords) && !multiSelectedCount()) {
      multiSelected = { none: true };
      updateMultiUI(q);
      showHeard('都没有');
      ackAndNext('哎，好嘞，我记下了。');
      return;
    }
    var matchedAny = false;
    q.options.forEach(function (opt) {
      if (opt.value === 'none') return;
      if (SA.voice.matchAny(result, opt.keywords)) {
        multiSelected[opt.value] = true;
        delete multiSelected.none;
        matchedAny = true;
      }
    });
    if (matchedAny) {
      updateMultiUI(q);
      var chosen = Object.keys(multiSelected).map(function (v) { return labelOf(q, v); });
      showHeard(chosen.join('、'));
      SA.voice.speak('我给您记下了' + chosen.join('和') + '。您要是还有别的病，接着说就行；没有的话，点下面的确认就好。');
    } else {
      SA.voice.speak('没听清，您说出病名就行，或者点下面的按钮。').then(function () { autoListen(); });
    }
  }

  function multiSelectedCount() {
    return Object.keys(multiSelected).filter(function (k) { return k !== 'none'; }).length;
  }

  /* ---------- 暴露与绑定 ---------- */
  SA.health = {
    start: start,
    questions: questions,
    getCurrentIndex: function () { return current; },
    _matchOption: matchOption          // 供自动化测试使用
  };
  SA.startInterview = start;

  function bind(id, handler) {
    var el = SA.$ && SA.$(id);
    if (el) el.addEventListener('click', handler);
  }

  document.addEventListener('DOMContentLoaded', function () {
    if (!SA.$) return;                 // 测试环境无 app.js 时跳过 DOM 绑定
    bind('btn-speak-question', function () {
      var q = questions[current];
      if (q && SA.voice) SA.voice.speak(q.spoken || q.question);
    });
    bind('btn-mic', function () {
      if (listening) return;
      if (SA.voice) SA.voice.stopSpeak();
      startListen();
    });
    bind('btn-skip-voice', function () {
      stopListenUI();
      if (SA.voice) SA.voice.speak('您点下面的按钮，选您的答案就好。');
      var box = SA.$('interview-answers');
      if (box) {
        box.style.outline = '4px solid var(--accent)';
        setTimeout(function () { box.style.outline = ''; }, 1600);
      }
    });
  });
})();