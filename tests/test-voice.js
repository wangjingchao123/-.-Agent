/* 语音模块自动化测试（无头浏览器安全：不依赖真实音频/麦克风） */
(function () {
  'use strict';
  var test = SATest.test;
  var testAsync = SATest.testAsync;
  var assert = SATest.assert;

  test('SA.voice 模块已加载', function () {
    assert(SA && SA.voice, 'SA.voice 未定义');
  });

  test('TTS/STT 支持检测返回布尔值', function () {
    assert(typeof SA.voice.isTTSSupported() === 'boolean', 'isTTSSupported 应返回布尔值');
    assert(typeof SA.voice.isSTTSupported() === 'boolean', 'isSTTSupported 应返回布尔值');
  });

  test('setEnabled(false) 后 speak 立即返回且不播报', function () {
    SA.voice.setEnabled(true);
    SA.voice.setEnabled(false);
    assert(SA.voice.enabled === false, 'enabled 应为 false');
    var p = SA.voice.speak('测试');
    assert(p && typeof p.then === 'function', 'speak 应返回 Promise');
    SA.voice.setEnabled(true); // 恢复
    assert(SA.voice.enabled === true, '恢复后 enabled 应为 true');
  });

  testAsync('speak 空字符串立即完成', function () {
    return SA.voice.speak('').then(function () {
      assert(true, '空字符串应直接 resolve');
    });
  });

  test('speak 返回 Promise', function () {
    var p = SA.voice.speak('你好');
    assert(p && typeof p.then === 'function', '应返回 Promise 对象');
    // 不等待真实播报结束（无头环境无音频设备），仅验证接口契约
  });

  test('matchAny 命中任一关键词返回 true', function () {
    var result = { text: '我今年七十岁', alternatives: [{ text: '我今年七十岁' }, { text: '我今年 Seventy 岁' }] };
    assert(SA.voice.matchAny(result, ['七十', '70']) === true, '应命中"七十"');
    assert(SA.voice.matchAny(result, ['八十']) === false, '不应命中"八十"');
  });

  test('matchAny 支持候选文本容错匹配', function () {
    // 模拟方言/近音：主结果听错，候选里有正确词
    var result = { text: '五八十三', alternatives: [{ text: '五十八' }, { text: '五十八岁' }] };
    assert(SA.voice.matchAny(result, ['五十八']) === true, '应从候选中命中"五十八"');
  });

  test('matchAny 对空结果返回 false', function () {
    assert(SA.voice.matchAny(null, ['你好']) === false, 'null 应返回 false');
    assert(SA.voice.matchAny({ text: '', alternatives: [] }, ['你好']) === false, '空文本应返回 false');
  });

  test('stopSpeak 不抛异常', function () {
    SA.voice.stopSpeak();
    assert(SA.voice.speaking === false, 'speaking 应为 false');
  });
})();