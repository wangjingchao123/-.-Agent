/* 方案生成模块自动化测试（纯逻辑，无头安全） */
(function () {
  'use strict';
  var test = SATest.test;
  var assert = SATest.assert;

  function ids(plan) { return plan.map(function (e) { return e.id; }); }
  function has(arr, v) { return arr.indexOf(v) !== -1; }

  test('SA.planner 模块已加载', function () {
    assert(SA && SA.planner, 'SA.planner 未定义');
    assert(typeof SA.planner.generatePlan === 'function', 'generatePlan 应为函数');
  });

  test('动作库结构完整', function () {
    assert(SA.planner.exercises.length >= 12, '动作库应至少12个动作，实际 ' + SA.planner.exercises.length);
    SA.planner.exercises.forEach(function (ex) {
      assert(typeof ex.id === 'string' && ex.id, '应有 id');
      assert(typeof ex.name === 'string' && ex.name, ex.id + ' 应有 name');
      assert(['坐位操', '平衡训练', '简化八段锦'].indexOf(ex.category) !== -1, ex.id + ' category 非法');
      assert(['low', 'mid'].indexOf(ex.intensity) !== -1, ex.id + ' intensity 非法');
      assert(['poor', 'assist', 'good'].indexOf(ex.minStanding) !== -1, ex.id + ' minStanding 非法');
      assert(typeof ex.desc === 'string' && ex.desc, ex.id + ' 应有 desc');
      assert(Array.isArray(ex.steps) && ex.steps.length >= 2, ex.id + ' 应有 steps');
      assert(typeof ex.tip === 'string', ex.id + ' 应有 tip');
      assert(ex.metric && (ex.metric.type === 'count' || ex.metric.type === 'seconds'), ex.id + ' metric 非法');
      assert(Array.isArray(ex.contraindications), ex.id + ' 应有 contraindications 数组');
    });
  });

  test('高血压：过滤转头/低头类动作', function () {
    var plan = SA.planner.generatePlan({ standing: 'good', hypertension: true });
    var p = ids(plan);
    assert(!has(p, 'bdt-houwang'), '五劳七伤往后瞧(转头)应被过滤');
    assert(!has(p, 'bdt-yaotou'), '摇头摆尾(低头转腰)应被过滤');
    assert(!has(p, 'bdt-panzu'), '双手攀足(低头弯腰)应被过滤');
    assert(has(p, 'bdt-tian'), '两手托天(安全)应保留');
  });

  test('膝关节炎：过滤起坐与马步类动作', function () {
    var plan = SA.planner.generatePlan({ standing: 'good', kneeArthritis: true });
    var p = ids(plan);
    assert(!has(p, 'chair-stand'), '扶椅起坐应被过滤');
    assert(!has(p, 'bdt-zanquan'), '攒拳怒目(马步)应被过滤');
  });

  test('骨质疏松：过滤前弯与单腿承重类动作', function () {
    var plan = SA.planner.generatePlan({ standing: 'good', osteoporosis: true });
    var p = ids(plan);
    assert(!has(p, 'heel-toe'), '脚跟脚尖站立应被过滤');
    assert(!has(p, 'single-leg'), '单腿站立应被过滤');
    assert(!has(p, 'bdt-panzu'), '双手攀足(前弯)应被过滤');
  });

  test('跌倒史：过滤所有失衡风险动作', function () {
    var plan = SA.planner.generatePlan({ standing: 'good', fallHistory: true });
    var p = ids(plan);
    assert(!has(p, 'single-leg'), '单腿站立应被过滤');
    assert(!has(p, 'tandem-stand'), '并足闭眼站立应被过滤');
    assert(!has(p, 'heel-toe'), '脚跟脚尖站立应被过滤');
    assert(has(p, 'seated-neck'), '坐位安全动作应保留');
  });

  test('站立能力差(poor)：方案仅含坐位操', function () {
    var plan = SA.planner.generatePlan({ standing: 'poor' });
    assert(plan.length > 0, '应至少有1个动作');
    plan.forEach(function (ex) {
      assert(ex.category === '坐位操', 'poor 站立不应出现 ' + ex.category + ' 动作');
    });
  });

  test('需扶靠站立(assist)：不出现 good 级动作', function () {
    var plan = SA.planner.generatePlan({ standing: 'assist' });
    var p = ids(plan);
    assert(!has(p, 'single-leg'), '单腿站立(good级)应被过滤');
    assert(!has(p, 'tandem-stand'), '并足闭眼(good级)应被过滤');
    assert(has(p, 'chair-stand'), '扶椅起坐(assist级)应保留');
  });

  test('80岁以上：仅保留低强度动作', function () {
    var plan = SA.planner.generatePlan({ standing: 'good', ageGroup: '80s' });
    assert(plan.length > 0, '应至少有1个动作');
    plan.forEach(function (ex) {
      assert(ex.intensity === 'low', '80岁以上不应出现中强度动作 ' + ex.name);
    });
  });

  test('健康老人：方案含三类动作且数量合理', function () {
    var plan = SA.planner.generatePlan({ standing: 'good', ageGroup: '60s', fallHistory: false });
    var cats = {};
    plan.forEach(function (ex) { cats[ex.category] = true; });
    assert(plan.length >= 6 && plan.length <= 9, '方案应6-9个动作，实际 ' + plan.length);
    assert(cats['坐位操'], '应含坐位操');
    assert(cats['平衡训练'], '应含平衡训练');
    assert(cats['简化八段锦'], '应含简化八段锦');
  });

  test('describeTags 健康标签文字化', function () {
    var tags = SA.planner.describeTags({ ageGroup: '70s', hypertension: true, standing: 'assist' });
    assert(has(tags, '70-79岁'), '应含年龄段');
    assert(has(tags, '高血压'), '应含高血压');
    assert(has(tags, '需扶靠站立'), '应含站立能力');
    var empty = SA.planner.describeTags({});
    assert(empty.length === 0, '空标签应返回空数组');
  });
})();