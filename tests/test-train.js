/* 训练模块测试：calcTarget / applyFeeling / renderSummary 容错 / 模块导出 */
(function () {
  var ex8 = { metric: { type: 'count', value: 8 } };
  var ex60 = { metric: { type: 'seconds', value: 60 } };

  SATest.test('train 模块加载', function () {
    SATest.assert(SA.train, 'SA.train 未定义');
    SATest.assert(typeof SA.train.calcTarget === 'function', 'calcTarget 未导出');
    SATest.assert(typeof SA.train.applyFeeling === 'function', 'applyFeeling 未导出');
    SATest.assert(typeof SA.startTraining === 'function', 'startTraining 别名未导出');
  });

  SATest.test('calcTarget 计数型难度1.0', function () {
    SATest.assertEqual(SA.train.calcTarget(ex8, 1), 8, '8次*1.0应=8');
  });

  SATest.test('calcTarget 计数型难度0.7向上取整', function () {
    SATest.assertEqual(SA.train.calcTarget(ex8, 0.7), 6, '8*0.7=5.6应向上取整为6');
  });

  SATest.test('calcTarget 计时型难度1.0', function () {
    SATest.assertEqual(SA.train.calcTarget(ex60, 1), 60, '60秒*1.0应=60');
  });

  SATest.test('calcTarget 计时型难度0.7取整', function () {
    SATest.assertEqual(SA.train.calcTarget(ex60, 0.7), 42, '60*0.7应=42');
  });

  SATest.test('calcTarget 缺少metric返回0', function () {
    SATest.assertEqual(SA.train.calcTarget({ name: 'x' }, 1), 0, '无metric应返回0');
    SATest.assertEqual(SA.train.calcTarget(null, 1), 0, 'null应返回0');
  });

  SATest.test('calcTarget 未传难度默认1.0', function () {
    SATest.assertEqual(SA.train.calcTarget(ex8), 8, 'undefined难度应按1.0计算');
  });

  SATest.test('applyFeeling good 难度不变', function () {
    var r = SA.train.applyFeeling(1, 'good');
    SATest.assertEqual(r.difficulty, 1, 'good难度不变');
    SATest.assert(!r.stop, 'good不应停止');
  });

  SATest.test('applyFeeling tired 难度降至0.7倍', function () {
    var r = SA.train.applyFeeling(1, 'tired');
    SATest.assertEqual(r.difficulty, 0.7, 'tired应降至0.7');
    SATest.assert(!r.stop, 'tired不应停止');
  });

  SATest.test('applyFeeling 连续tired正常衰减', function () {
    var r = SA.train.applyFeeling(SA.train.applyFeeling(1, 'tired').difficulty, 'tired');
    SATest.assertEqual(r.difficulty, 0.49, '两次tired应=0.49');
  });

  SATest.test('applyFeeling tired 难度下限0.4', function () {
    var r = SA.train.applyFeeling(0.5, 'tired');
    SATest.assertEqual(r.difficulty, 0.4, '0.5*0.7=0.35应保底0.4');
  });

  SATest.test('applyFeeling dizzy 立即停止', function () {
    var r = SA.train.applyFeeling(1, 'dizzy');
    SATest.assert(r.stop, 'dizzy应停止');
    SATest.assertEqual(r.difficulty, 1, 'dizzy难度不变');
  });

  SATest.test('renderSummary 无DOM元素时不抛错', function () {
    SA.train.renderSummary();
    SATest.assert(true, '执行无异常即通过');
  });
})();