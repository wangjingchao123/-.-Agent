/* 银龄安动 - 语音交互模块（TTS 播报 + STT 识别封装）
 * 特点：
 *  - 播报语速适中偏慢、音调柔和，优先微软晓晓音色，适合老年人
 *  - 长文本按句拆分逐句播报，句间自然停顿，去机器人感
 *  - 识别返回多个候选文本，方便方言/近音容错匹配
 *  - 支持普通话 zh-CN / 粤语 zh-HK / 台湾普通话 zh-TW
 *  - 不支持语音时优雅降级为纯按钮交互
 */
(function () {
  'use strict';
  window.SA = window.SA || {};

  var synth = window.speechSynthesis || null;
  var RecognitionCtor = window.SpeechRecognition || window.webkitSpeechRecognition || null;
  var currentRec = null;   // 当前识别实例，用于 abort

  var voice = {
    enabled: true,
    speaking: false,
    listening: false
  };

  /* ---------- 开关 ---------- */
  voice.setEnabled = function (on) {
    voice.enabled = !!on;
    if (!on) {
      speakSeq++;   // 使进行中的播报链失效
      if (synth) { try { synth.cancel(); } catch (e) {} }
      voice.speaking = false;
    }
  };

  voice.isTTSSupported = function () { return !!synth; };
  voice.isSTTSupported = function () { return !!RecognitionCtor; };

  /* ---------- 中文语音预热（getVoices 首次可能为空） ---------- */
  var zhVoices = [];
  function refreshVoices() {
    if (!synth) return;
    var all = synth.getVoices() || [];
    zhVoices = all.filter(function (v) { return v.lang && v.lang.toLowerCase().indexOf('zh') === 0; });
  }
  if (synth) {
    refreshVoices();
    if (typeof synth.onvoiceschanged !== 'undefined') {
      synth.onvoiceschanged = refreshVoices;
    }
  }

  /* 优先微软晓晓（最自然的中文女声），其次其他微软中文女声，再次任一中文语音 */
  function pickVoice(lang) {
    if (!zhVoices.length) refreshVoices();
    if (!zhVoices.length) return null;
    var pool = zhVoices;
    var exact = pool.filter(function (v) { return v.lang.toLowerCase().indexOf(String(lang).toLowerCase()) === 0; });
    if (exact.length) pool = exact;
    var xiaoxiao = pool.filter(function (v) { return /xiaoxiao|晓晓/i.test(v.name); });
    if (xiaoxiao.length) return xiaoxiao[0];
    var msFemale = pool.filter(function (v) { return /microsoft|微软/i.test(v.name) && !/kangkang|康康/i.test(v.name); });
    if (msFemale.length) return msFemale[0];
    return pool[0];
  }

  /* ---------- 句间停顿：按标点拆句 ---------- */
  /* 在 。！？；处断长句，在 ，,处断短句；播报时分别加长/短停顿 */
  function splitSentences(text) {
    return String(text)
      .replace(/([。！？；])/g, '$1\u0001')
      .replace(/([，,])/g, '$1\u0002')
      .split(/[\u0001\u0002]/)
      .map(function (s) { return s.trim(); })
      .filter(Boolean);
  }

  /* 单句播报 */
  function speakOne(text, lang, opts) {
    return new Promise(function (resolve) {
      try { synth.cancel(); } catch (e) {}
      var u = new SpeechSynthesisUtterance(String(text));
      u.lang = lang;
      u.rate = opts.rate || 0.94;    // 语速：原0.75×1.25，自然流畅仍清晰
      u.pitch = opts.pitch || 0.95;  // 音调略降，更柔和自然
      u.volume = opts.volume || 1.0;
      var v = pickVoice(lang);
      if (v) u.voice = v;
      var done = false;
      function finish() { if (!done) { done = true; voice.speaking = false; resolve(); } }
      u.onend = finish;
      u.onerror = finish;
      voice.speaking = true;
      synth.speak(u);
      setTimeout(finish, 12000);   // 单句兜底
    });
  }

  /* ---------- 语音播报（TTS） ---------- */
  /* speak(text, opts) -> Promise，长文本自动拆句逐句播报，句间加自然停顿 */
  var speakSeq = 0;   // 播报序号，新播报/打断会使旧链失效
  voice.speak = function (text, opts) {
    opts = opts || {};
    if (!voice.enabled || !synth || !text) return Promise.resolve();
    var parts = splitSentences(text);
    if (!parts.length) return Promise.resolve();
    var mySeq = ++speakSeq;
    var lang = opts.lang || (SA.state ? SA.state.lang : 'zh-CN') || 'zh-CN';
    var longPause = opts.pause != null ? opts.pause : 420;         // 句号后停顿
    var shortPause = opts.commaPause != null ? opts.commaPause : 220; // 逗号后短停顿
    var chain = Promise.resolve();
    parts.forEach(function (part) {
      chain = chain.then(function () {
        if (mySeq !== speakSeq) return;   // 已被新播报或打断取代
        return speakOne(part, lang, opts);
      }).then(function () {
        if (mySeq !== speakSeq) return;   // 已被打断，不再继续
        return new Promise(function (r) {
          setTimeout(r, /[。！？；]$/.test(part) ? longPause : shortPause);
        });
      });
    });
    return chain;
  };

  /* 打断当前播报 */
  voice.stopSpeak = function () {
    speakSeq++;   // 使进行中的播报链失效，句间停顿后不再继续
    if (synth) { try { synth.cancel(); } catch (e) {} }
    voice.speaking = false;
  };

  /* 中止当前语音识别 */
  voice.abortListen = function () {
    if (currentRec) { try { currentRec.abort(); } catch (e) {} currentRec = null; }
    voice.listening = false;
  };

  /* ---------- 语音识别（STT） ---------- */
  /* listen(opts) -> Promise<{text, alternatives}> */
  voice.listen = function (opts) {
    opts = opts || {};
    return new Promise(function (resolve, reject) {
      if (!RecognitionCtor) { reject(new Error('当前浏览器不支持语音识别，请点按按钮回答')); return; }
      if (!voice.enabled) { reject(new Error('语音已关闭')); return; }
      var r = new RecognitionCtor();
      currentRec = r;
      var lang = opts.lang || (SA.state ? SA.state.lang : 'zh-CN') || 'zh-CN';
      r.lang = lang;
      r.continuous = false;
      r.interimResults = false;
      r.maxAlternatives = 4;   // 多候选用于方言/近音容错
      var settled = false;
      function clearRec() { if (currentRec === r) currentRec = null; }
      r.onresult = function (e) {
        if (settled) return;
        settled = true;
        clearRec();
        var res = e.results[0];
        var alts = [];
        for (var i = 0; i < res.length && i < 4; i++) {
          alts.push({ text: res[i].transcript, confidence: res[i].confidence || 0 });
        }
        voice.listening = false;
        resolve({ text: alts[0] ? alts[0].text : '', alternatives: alts });
      };
      r.onerror = function (e) {
        if (settled) return;
        settled = true;
        clearRec();
        voice.listening = false;
        var msg = '语音识别失败';
        if (e.error === 'not-allowed' || e.error === 'service-not-allowed') msg = '麦克风未授权，请点按按钮回答';
        else if (e.error === 'no-speech') msg = '没有听清，请再试一次或点按按钮';
        reject(new Error(msg));
      };
      r.onend = function () {
        if (settled) return;
        settled = true;
        clearRec();
        voice.listening = false;
        resolve({ text: '', alternatives: [] });
      };
      try {
        voice.listening = true;
        r.start();
      } catch (err) {
        clearRec();
        voice.listening = false;
        reject(err);
      }
    });
  };

  /* ---------- 方言/近音容错匹配 ---------- */
  /* 在多个候选文本里查找是否命中关键词组（任一命中即返回 true） */
  voice.matchAny = function (result, keywords) {
    if (!result) return false;
    var texts = [];
    if (result.text) texts.push(result.text);
    if (result.alternatives) result.alternatives.forEach(function (a) { if (a.text) texts.push(a.text); });
    return keywords.some(function (kw) {
      return texts.some(function (t) { return t.indexOf(kw) !== -1; });
    });
  };

  SA.voice = voice;
  console.log('[语音模块] TTS支持:', voice.isTTSSupported(), 'STT支持:', voice.isSTTSupported());
})();