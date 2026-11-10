/* 银龄安动 - 姿态判断模块（纯函数，无副作用，可独立单测）
 * 基于 MediaPipe Pose 33 关键点归一化坐标 {x,y,z,visibility}
 * 关键点索引：11左肩 12右肩 23左髋 24右髋 25左膝 26右膝 27左踝 28右踝
 */
(function () {
  'use strict';
  window.SA = window.SA || {};

  var LM = { LSH: 11, RSH: 12, LHIP: 23, RHIP: 24, LKNEE: 25, RKNEE: 26, LANKLE: 27, RANKLE: 28 };

  /* 三点夹角（度）：顶点在 b */
  function angle(a, b, c) {
    if (!a || !b || !c) return null;
    var v1 = { x: a.x - b.x, y: a.y - b.y };
    var v2 = { x: c.x - b.x, y: c.y - b.y };
    var dot = v1.x * v2.x + v1.y * v2.y;
    var m1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y);
    var m2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y);
    if (m1 === 0 || m2 === 0) return null;
    var cos = dot / (m1 * m2);
    cos = Math.max(-1, Math.min(1, cos));
    return Math.acos(cos) * 180 / Math.PI;
  }

  function mid(p1, p2) {
    return { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2, visibility: Math.min(p1.visibility || 1, p2.visibility || 1) };
  }

  function visible(p) { return p && (p.visibility === undefined || p.visibility > 0.5); }

  /* 躯干直立检测：肩中点-髋中点连线与垂直方向夹角（度），>25° 视为前倾 */
  function trunkTilt(landmarks) {
    if (!landmarks) return null;
    var ls = landmarks[LM.LSH], rs = landmarks[LM.RSH], lh = landmarks[LM.LHIP], rh = landmarks[LM.RHIP];
    if (!visible(ls) || !visible(rs) || !visible(lh) || !visible(rh)) return null;
    var s = mid(ls, rs), h = mid(lh, rh);
    var dx = h.x - s.x, dy = h.y - s.y;
    if (dy <= 0) return null;                     // 坐标系异常（髋在肩上方）
    return Math.atan2(Math.abs(dx), dy) * 180 / Math.PI;
  }

  /* 膝盖是否超过脚尖（侧面视角，归一化 x 差），超过阈值返回 true */
  function kneeOverToe(landmarks) {
    if (!landmarks) return null;
    var knee = landmarks[LM.LKNEE], ankle = landmarks[LM.LANKLE];
    if (!visible(knee) || !visible(ankle)) return null;
    return (knee.x - ankle.x) > 0.05;
  }

  /* 膝关节角度（左腿：髋-膝-踝），用于下蹲深度判断 */
  function kneeAngle(landmarks) {
    if (!landmarks) return null;
    var hip = landmarks[LM.LHIP], knee = landmarks[LM.LKNEE], ankle = landmarks[LM.LANKLE];
    if (!visible(hip) || !visible(knee) || !visible(ankle)) return null;
    return angle(hip, knee, ankle);
  }

  /* 是否单腿支撑（左右踝 y 差大于阈值，抬腿侧 y 更小） */
  function isSingleLeg(landmarks) {
    if (!landmarks) return null;
    var la = landmarks[LM.LANKLE], ra = landmarks[LM.RANKLE];
    if (!visible(la) || !visible(ra)) return null;
    return Math.abs(la.y - ra.y) > 0.08;
  }

  /* 两点平面距离（归一化坐标） */
  function dist2d(a, b) {
    var dx = a.x - b.x, dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  /* OK 手势检测（👌）：拇指尖与食指尖捏合成圆，小指伸展远离拇指
   * 用于训练前"等待确认开始"阶段，识别到即启动正式训练
   * 关键点：15左腕 16右腕 17左小指 18右小指 19左食指 20右食指 21左拇指 22右拇指
   * 阈值说明：以腕-小指距离为手长参考 scale
   *  - 拇指-食指距离 < 0.3×scale（捏合）
   *  - 拇指-小指距离 > 0.5×scale（其余手指伸展，排除握拳）
   */
  function detectOKGesture(landmarks) {
    if (!landmarks) return false;
    var hands = [
      { w: landmarks[16], t: landmarks[22], i: landmarks[20], p: landmarks[18] },
      { w: landmarks[15], t: landmarks[21], i: landmarks[19], p: landmarks[17] }
    ];
    for (var k = 0; k < hands.length; k++) {
      var h = hands[k];
      if (!h.w || !h.t || !h.i || !h.p) continue;
      if (!visible(h.w) || !visible(h.t) || !visible(h.i) || !visible(h.p)) continue;
      var scale = dist2d(h.w, h.p);
      if (scale < 0.03) continue;                 // 手太小/太远，关键点不可靠
      var dTI = dist2d(h.t, h.i);                 // 拇指-食指（捏合应很小）
      var dTP = dist2d(h.t, h.p);                 // 拇指-小指（伸展应较大）
      if (dTI < scale * 0.3 && dTP > scale * 0.5) return true;
    }
    return false;
  }

  /* 综合姿势评估：返回 { ok, messages:[语音提醒文本] }
   * 规则按动作类别启用，避免对坐位操误报站立类问题 */
  function evaluatePosture(landmarks, exercise) {
    var messages = [];
    if (!landmarks || !exercise) return { ok: true, messages: messages };

    /* 躯干前倾：所有动作通用 */
    var tilt = trunkTilt(landmarks);
    if (tilt !== null && tilt > 25) {
      messages.push('注意挺直腰背，不要前倾');
    }

    /* 站立类动作：膝盖过脚尖检测 */
    if (exercise.category === '平衡训练' || (exercise.id === 'bdt-zanquan')) {
      var over = kneeOverToe(landmarks);
      if (over === true) {
        messages.push('膝盖不要超过脚尖，往后坐一点');
      }
    }

    /* 下蹲类（马步/起坐）：膝角度过小提示起身 */
    if (exercise.id === 'bdt-zanquan' || exercise.id === 'chair-stand' || exercise.id === 'bdt-yaotou') {
      var ka = kneeAngle(landmarks);
      if (ka !== null && ka < 95) {
        messages.push('蹲得有点低，稍微起来一点');
      }
    }

    /* 单腿站立：未检测到抬腿则提醒 */
    if (exercise.id === 'single-leg') {
      var sl = isSingleLeg(landmarks);
      if (sl === false) {
        messages.push('请把一条腿抬起来，手扶稳椅背');
      }
    }

    return { ok: messages.length === 0, messages: messages };
  }

  SA.pose = {
    LM: LM,
    angle: angle,
    trunkTilt: trunkTilt,
    kneeOverToe: kneeOverToe,
    kneeAngle: kneeAngle,
    isSingleLeg: isSingleLeg,
    detectOKGesture: detectOKGesture,
    evaluatePosture: evaluatePosture
  };
})();