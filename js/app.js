/* 银龄安动 - 主控与屏幕路由 */
(function () {
  'use strict';
  window.SA = window.SA || {};

  /* ---------- 全局状态 ---------- */
  var state = {
    screen: 'home',      // 当前屏幕
    health: null,        // 健康标签对象
    plan: [],            // 个性化方案（动作数组）
    trainIndex: 0,       // 当前训练动作序号
    voiceOn: true,       // 语音总开关
    lang: 'zh-CN'        // 语音识别语言（可切方言/粤语）
  };
  SA.state = state;

  /* ---------- 屏幕路由 ---------- */
  var screens = ['home', 'interview', 'plan', 'train', 'summary'];

  SA.go = function (name) {
    if (screens.indexOf(name) === -1) { console.warn('[路由] 未知屏幕:', name); return; }
    screens.forEach(function (s) {
      var el = document.getElementById('screen-' + s);
      if (el) el.classList.toggle('active', s === name);
    });
    state.screen = name;
    window.scrollTo(0, 0);
    if (typeof SA.onScreenChange === 'function') SA.onScreenChange(name);
  };

  SA.$ = function (id) { return document.getElementById(id); };

  /* ---------- 页面初始化 ---------- */
  document.addEventListener('DOMContentLoaded', function () {
    /* 返回按钮 */
    document.querySelectorAll('[data-back]').forEach(function (b) {
      b.addEventListener('click', function () { SA.go(b.getAttribute('data-back')); });
    });

    /* 首页：开始问询 */
    SA.$('btn-start').addEventListener('click', function () {
      if (typeof SA.startInterview === 'function') SA.startInterview();
      else SA.go('interview');
    });

    /* 语音总开关 */
    SA.$('btn-voice-set').addEventListener('click', function () {
      state.voiceOn = !state.voiceOn;
      this.textContent = state.voiceOn ? '🔊 语音：开' : '🔇 语音：关';
      if (SA.voice && typeof SA.voice.setEnabled === 'function') SA.voice.setEnabled(state.voiceOn);
    });

    /* 方案页按钮（后续模块会覆盖更完整逻辑） */
    SA.$('btn-start-train').addEventListener('click', function () {
      if (typeof SA.startTraining === 'function') SA.startTraining();
      else SA.go('train');
    });
    SA.$('btn-replan').addEventListener('click', function () { SA.go('interview'); if (SA.startInterview) SA.startInterview(); });
    SA.$('btn-again').addEventListener('click', function () { SA.go('train'); if (SA.startTraining) SA.startTraining(); });
    SA.$('btn-home').addEventListener('click', function () { SA.go('home'); });

    console.log('[银龄安动] 应用已加载，当前屏幕: home');
  });
})();