/* 健康问询模块自动化测试（无头安全：仅测数据结构与匹配逻辑，不触发真实语音/DOM流程） */
(function () {
  'use strict';
  var test = SATest.test;
  var assert = SATest.assert;

  test('SA.health 模块已加载', function () {
    assert(SA && SA.health, 'SA.health 未定义');
    assert(typeof SA.startInterview === 'function', 'SA.startInterview 应为函数');
  });

  test('问询题目共 5 题', function () {
    assert(SA.health.questions.length === 5, '应有 5 道题，实际 ' + SA.health.questions.length);
  });

  test('题目结构完整（id/type/question/options）', function () {
    var ids = ['age', 'diseases', 'pain', 'fall', 'standing'];
    SA.health.questions.forEach(function (q, i) {
      assert(q.id === ids[i], '第' + (i + 1) + '题 id 应为 ' + ids[i] + '，实际 ' + q.id);
      assert(q.type === 'single' || q.type === 'multi', '第' + (i + 1) + '题 type 应为 single/multi');
      assert(typeof q.question === 'string' && q.question.length > 0, '第' + (i + 1) + '题应有 question 文本');
      assert(Array.isArray(q.options) && q.options.length >= 2, '第' + (i + 1) + '题 options 至少 2 项');
      q.options.forEach(function (opt) {
        assert(typeof opt.value === 'string', '选项应含 value');
        assert(typeof opt.label === 'string', '选项应含 label');
        assert(Array.isArray(opt.keywords) && opt.keywords.length > 0, '选项应含 keywords 数组');
        assert(opt.tag && typeof opt.tag === 'object', '选项应含 tag 对象');
      });
    });
  });

  test('慢性病题含高血压/膝关节炎/骨质疏松/糖尿病标签', function () {
    var q = SA.health.questions.filter(function (x) { return x.id === 'diseases'; })[0];
    var vals = q.options.map(function (o) { return o.value; });
    ['hypertension', 'kneeArthritis', 'osteoporosis', 'diabetes', 'none'].forEach(function (v) {
      assert(vals.indexOf(v) !== -1, '应含选项 ' + v);
    });
  });

  test('_matchOption 单选题：关键词命中正确选项', function () {
    var ageQ = SA.health.questions.filter(function (x) { return x.id === 'age'; })[0];
    var m = SA.health._matchOption(ageQ, { text: '我今年七十二岁', alternatives: [] });
    assert(m && m.value === '70s', '应命中 70s，实际 ' + (m && m.value));
  });

  test('_matchOption 否定选项排后：\"没有高血压\"命中 none', function () {
    var q = SA.health.questions.filter(function (x) { return x.id === 'diseases'; })[0];
    var m = SA.health._matchOption(q, { text: '我没有高血压', alternatives: [] });
    assert(m && m.value === 'none', '应命中 none，实际 ' + (m && m.value));
  });

  test('_matchOption 候选文本容错：方言近音从 alternatives 命中', function () {
    var q = SA.health.questions.filter(function (x) { return x.id === 'pain'; })[0];
    var result = { text: '膝盖听不清', alternatives: [{ text: '膝盖疼' }] };
    var m = SA.health._matchOption(q, result);
    assert(m && m.value === 'kneePain', '应从候选命中 kneePain，实际 ' + (m && m.value));
  });

  test('_matchOption 无匹配返回 null', function () {
    var q = SA.health.questions.filter(function (x) { return x.id === 'fall'; })[0];
    var m = SA.health._matchOption(q, { text: '今天天气不错', alternatives: [] });
    assert(m === null, '无匹配应返回 null，实际 ' + m);
  });

  test('_matchOption 跌倒题：\"摔了一跤\"命中跌倒过', function () {
    var q = SA.health.questions.filter(function (x) { return x.id === 'fall'; })[0];
    var m = SA.health._matchOption(q, { text: '去年摔了一跤', alternatives: [] });
    assert(m && m.value === 'yes', '应命中 yes，实际 ' + (m && m.value));
  });
})();