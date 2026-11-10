/* 姿态判断模块自动化测试（合成关键点数据，无头安全） */
(function () {
  'use strict';
  var test = SATest.test;
  var assert = SATest.assert;

  /* 构造33点合成关键点：默认直立站姿 */
  function mk(overrides) {
    var l = [];
    for (var i = 0; i < 33; i++) l.push({ x: 0.5, y: 0.5, z: 0, visibility: 1 });
    l[11] = { x: 0.45, y: 0.30, visibility: 1 };  // 左肩
    l[12] = { x: 0.55, y: 0.30, visibility: 1 };  // 右肩
    l[23] = { x: 0.47, y: 0.55, visibility: 1 };  // 左髋
    l[24] = { x: 0.53, y: 0.55, visibility: 1 };  // 右髋
    l[25] = { x: 0.46, y: 0.75, visibility: 1 };  // 左膝
    l[26] = { x: 0.54, y: 0.75, visibility: 1 };  // 右膝
    l[27] = { x: 0.45, y: 0.95, visibility: 1 };  // 左踝
    l[28] = { x: 0.55, y: 0.95, visibility: 1 };  // 右踝
    if (overrides) Object.keys(overrides).forEach(function (k) { l[+k] = overrides[k]; });
    return l;
  }

  test('SA.pose 模块已加载', function () {
    assert(SA && SA.pose, 'SA.pose 未定义');
  });

  test('angle() 直角与平角计算', function () {
    var r = SA.pose.angle({ x: 0, y: 0 }, { x: 0, y: 1 }, { x: 1, y: 1 });
    assert(Math.abs(r - 90) < 1, '直角应为90度，实际 ' + r);
    var flat = SA.pose.angle({ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 2, y: 0 });
    assert(Math.abs(flat - 180) < 1, '平角应为180度，实际 ' + flat);
  });

  test('trunkTilt 直立躯干接近0度', function () {
    var t = SA.pose.trunkTilt(mk());
    assert(t !== null && t < 10, '直立应<10度，实际 ' + t);
  });

  test('trunkTilt 前倾躯干超过25度', function () {
    var t = SA.pose.trunkTilt(mk({
      23: { x: 0.65, y: 0.55, visibility: 1 },
      24: { x: 0.71, y: 0.55, visibility: 1 }
    }));
    assert(t !== null && t > 25, '前倾应>25度，实际 ' + t);
  });

  test('trunkTilt 关键点不可见返回null', function () {
    var t = SA.pose.trunkTilt(mk({ 23: { x: 0.47, y: 0.55, visibility: 0.1 } }));
    assert(t === null, '不可见应返回 null');
  });

  test('kneeOverToe 膝盖超过脚尖返回true', function () {
    var r = SA.pose.kneeOverToe(mk({ 25: { x: 0.55, y: 0.75, visibility: 1 } }));
    assert(r === true, '膝过脚尖应为 true，实际 ' + r);
    var ok = SA.pose.kneeOverToe(mk());
    assert(ok === false, '正常应为 false，实际 ' + ok);
  });

  test('kneeAngle 直立接近180度，深蹲小于95度', function () {
    var up = SA.pose.kneeAngle(mk());
    assert(up > 160, '直立膝角应>160，实际 ' + up);
    var squat = SA.pose.kneeAngle(mk({
      23: { x: 0.50, y: 0.72, visibility: 1 },
      25: { x: 0.58, y: 0.75, visibility: 1 },
      27: { x: 0.42, y: 0.95, visibility: 1 }
    }));
    assert(squat < 95, '深蹲膝角应<95，实际 ' + squat);
  });

  test('isSingleLeg 抬腿检测', function () {
    var r = SA.pose.isSingleLeg(mk({ 27: { x: 0.45, y: 0.70, visibility: 1 } }));
    assert(r === true, '抬左腿应为 true，实际 ' + r);
    var both = SA.pose.isSingleLeg(mk());
    assert(both === false, '双脚着地应为 false，实际 ' + both);
  });

  test('evaluatePosture 前倾时给出挺直腰背提醒', function () {
    var ex = { id: 'seated-neck', category: '坐位操' };
    var r = SA.pose.evaluatePosture(mk({
      23: { x: 0.65, y: 0.55, visibility: 1 },
      24: { x: 0.71, y: 0.55, visibility: 1 }
    }), ex);
    assert(r.ok === false, '前倾应不通过');
    assert(r.messages.some(function (m) { return m.indexOf('挺直腰背') !== -1; }), '应含挺直腰背提醒');
  });

  test('evaluatePosture 坐位操不误报膝盖问题', function () {
    var ex = { id: 'seated-neck', category: '坐位操' };
    var r = SA.pose.evaluatePosture(mk({ 25: { x: 0.55, y: 0.75, visibility: 1 } }), ex);
    assert(r.messages.every(function (m) { return m.indexOf('脚尖') === -1; }), '坐位操不应报膝盖过脚尖');
  });

  test('evaluatePosture 姿势正确返回ok', function () {
    var ex = { id: 'bdt-tian', category: '简化八段锦' };
    var r = SA.pose.evaluatePosture(mk(), ex);
    assert(r.ok === true, '正确姿势应 ok:true');
    assert(r.messages.length === 0, '不应有提醒信息');
  });

  /* ---- OK 手势检测（训练前等待确认开始） ---- */
  /* 关键点：16右腕 18右小指 20右食指 22右拇指；15/17/19/21 为左侧对应 */
  test('detectOKGesture 右手OK手势识别', function () {
    assert(typeof SA.pose.detectOKGesture === 'function', 'detectOKGesture 未导出');
    /* 腕(0.6,0.5) 小指(0.7,0.5) → scale=0.1；拇指食指捏合 dTI=0.02<0.03；拇指小指伸展 dTP=0.08>0.05 */
    var ok = mk({
      16: { x: 0.60, y: 0.50, visibility: 1 },
      18: { x: 0.70, y: 0.50, visibility: 1 },
      20: { x: 0.62, y: 0.52, visibility: 1 },
      22: { x: 0.62, y: 0.50, visibility: 1 }
    });
    assert(SA.pose.detectOKGesture(ok) === true, '右手OK手势应识别为true');
  });

  test('detectOKGesture 左手OK手势识别', function () {
    var ok = mk({
      15: { x: 0.40, y: 0.50, visibility: 1 },
      17: { x: 0.30, y: 0.50, visibility: 1 },
      19: { x: 0.38, y: 0.52, visibility: 1 },
      21: { x: 0.38, y: 0.50, visibility: 1 }
    });
    assert(SA.pose.detectOKGesture(ok) === true, '左手OK手势应识别为true');
  });

  test('detectOKGesture 张开手掌不误判', function () {
    /* 拇指食指远离（dTI=0.1>0.03），非捏合 */
    var open = mk({
      16: { x: 0.60, y: 0.50, visibility: 1 },
      18: { x: 0.70, y: 0.50, visibility: 1 },
      20: { x: 0.65, y: 0.45, visibility: 1 },
      22: { x: 0.55, y: 0.45, visibility: 1 }
    });
    assert(SA.pose.detectOKGesture(open) === false, '张开手掌不应判为OK');
  });

  test('detectOKGesture 握拳不误判', function () {
    /* 拇指食指捏合但小指也收拢（dTP=0.02<0.05），排除握拳 */
    var fist = mk({
      16: { x: 0.60, y: 0.50, visibility: 1 },
      18: { x: 0.70, y: 0.50, visibility: 1 },
      20: { x: 0.68, y: 0.51, visibility: 1 },
      22: { x: 0.68, y: 0.50, visibility: 1 }
    });
    assert(SA.pose.detectOKGesture(fist) === false, '握拳不应判为OK');
  });

  test('detectOKGesture 关键点不可见返回false', function () {
    var inv = mk({
      16: { x: 0.60, y: 0.50, visibility: 0.1 },
      18: { x: 0.70, y: 0.50, visibility: 1 },
      20: { x: 0.62, y: 0.52, visibility: 1 },
      22: { x: 0.62, y: 0.50, visibility: 1 }
    });
    assert(SA.pose.detectOKGesture(inv) === false, '腕不可见应返回false');
  });

  test('detectOKGesture null与默认姿势返回false', function () {
    assert(SA.pose.detectOKGesture(null) === false, 'null应返回false');
    /* 默认mk()手部关键点全重叠，scale=0 不可靠，应返回false */
    assert(SA.pose.detectOKGesture(mk()) === false, '默认姿势不应判为OK');
  });
})();