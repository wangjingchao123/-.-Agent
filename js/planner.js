/* 银龄安动 - 方案生成模块
 * 动作库 + 健康标签高危过滤 + 个性化方案生成 + 方案页渲染
 * 过滤维度：禁忌症(contraindications) / 站立能力(minStanding) / 高龄降强度
 */
(function () {
  'use strict';
  window.SA = window.SA || {};

  /* 站立能力等级：poor(坐轮椅/卧床) < assist(需扶靠) < good(独立站稳) */
  var standingRank = { poor: 0, assist: 1, good: 2 };

  /* ---------- 动作库 ---------- */
  var exercises = [
    /* 坐位操（任何站立能力均可） */
    { id: 'seated-neck', name: '坐位颈部活动', category: '坐位操', intensity: 'low', minStanding: 'poor',
      desc: '坐直，缓慢低头、仰头、左转、右转，每个方向停留3秒',
      steps: ['坐直，双脚平放', '缓慢低头停留3秒', '缓慢仰头停留3秒', '左转右转各3秒'],
      tip: '动作要慢，不要猛甩头', metric: { type: 'count', value: 4 },
      contraindications: [] },
    { id: 'seated-shoulder', name: '坐位肩部环绕', category: '坐位操', intensity: 'low', minStanding: 'poor',
      desc: '坐直，双肩同时向前、向后画圈环绕，放松肩颈',
      steps: ['坐直放松', '双肩向前画圈8次', '双肩向后画圈8次'],
      tip: '幅度舒适即可，不耸肩憋气', metric: { type: 'count', value: 16 },
      contraindications: [] },
    { id: 'seated-ankle', name: '坐位踝泵运动', category: '坐位操', intensity: 'low', minStanding: 'poor',
      desc: '坐位，双脚脚尖用力上勾再下踩，促进下肢血液循环',
      steps: ['坐直，双脚着地', '脚尖用力上勾停留2秒', '脚尖下踩停留2秒'],
      tip: '对糖尿病老人尤其好，可多做', metric: { type: 'count', value: 20 },
      contraindications: [] },
    { id: 'seated-march', name: '坐位踏步', category: '坐位操', intensity: 'low', minStanding: 'poor',
      desc: '坐直，双腿交替抬起如踏步，双手自然摆动',
      steps: ['坐直，手扶椅面', '左腿抬起放下', '右腿抬起放下', '交替进行'],
      tip: '膝盖有不适就抬低一点', metric: { type: 'seconds', value: 60 },
      contraindications: [] },
    { id: 'seated-breath', name: '坐位深呼吸', category: '坐位操', intensity: 'low', minStanding: 'poor',
      desc: '坐直，鼻吸气4秒，口缓呼6秒，放松身心',
      steps: ['坐直，手放腹部', '鼻吸气4秒', '口缓呼6秒'],
      tip: '呼吸要缓，不要憋气', metric: { type: 'count', value: 6 },
      contraindications: [] },
    /* 平衡训练（需一定站立能力，跌倒史/骨质疏松过滤高危项） */
    { id: 'chair-stand', name: '扶椅起坐', category: '平衡训练', intensity: 'mid', minStanding: 'assist',
      desc: '站在稳固椅子前，扶椅背缓慢坐下再站起，锻炼下肢力量',
      steps: ['站在椅前，手扶椅背', '缓慢坐下', '缓慢站起'],
      tip: '椅子要靠墙固定，起立不要猛', metric: { type: 'count', value: 8 },
      contraindications: ['kneeArthritis', 'kneePain'] },
    { id: 'heel-toe', name: '脚跟脚尖站立', category: '平衡训练', intensity: 'mid', minStanding: 'assist',
      desc: '扶椅背，双脚脚跟抬起用脚尖站立，停留3秒后放下',
      steps: ['扶椅背站直', '脚跟抬起用脚尖站3秒', '缓慢放下'],
      tip: '一定扶稳椅背再抬脚跟', metric: { type: 'count', value: 8 },
      contraindications: ['fallHistory', 'osteoporosis'] },
    { id: 'single-leg', name: '扶椅单腿站立', category: '平衡训练', intensity: 'mid', minStanding: 'good',
      desc: '扶椅背，单腿站立10秒后换腿，锻炼平衡能力',
      steps: ['扶椅背站直', '左腿抬起站10秒', '换右腿站10秒'],
      tip: '晃动就马上扶稳，不要硬撑', metric: { type: 'count', value: 6 },
      contraindications: ['fallHistory', 'osteoporosis'] },
    { id: 'tandem-stand', name: '并足闭眼站立', category: '平衡训练', intensity: 'mid', minStanding: 'good',
      desc: '双脚并拢，先睁眼站稳，再尝试闭眼5秒，旁边要有依靠',
      steps: ['双脚并拢站直', '睁眼站稳10秒', '闭眼站5秒（手可扶椅）'],
      tip: '闭眼时手必须能随时扶到椅背', metric: { type: 'count', value: 4 },
      contraindications: ['fallHistory'] },
    /* 简化八段锦（站立位，按禁忌过滤） */
    { id: 'bdt-tian', name: '两手托天理三焦', category: '简化八段锦', intensity: 'low', minStanding: 'assist',
      desc: '站立，双手交叉上托过头顶，缓慢伸展全身',
      steps: ['站立，双手腹前交叉', '上托过顶，抬头看手', '缓慢放下'],
      tip: '上托时吸气，放下时呼气', metric: { type: 'count', value: 6 },
      contraindications: [] },
    { id: 'bdt-danju', name: '调理脾胃须单举', category: '简化八段锦', intensity: 'low', minStanding: 'assist',
      desc: '站立，一手上举一手下按，左右交替，调理脾胃',
      steps: ['站立，左手翻掌上举', '右手下按，对拉拔长', '左右交替'],
      tip: '动作舒展，力道柔和', metric: { type: 'count', value: 6 },
      contraindications: [] },
    { id: 'bdt-houwang', name: '五劳七伤往后瞧', category: '简化八段锦', intensity: 'low', minStanding: 'assist',
      desc: '站立，头缓慢转向一侧后方，眼随手动，两侧交替',
      steps: ['站立，双臂体侧', '头缓慢左转看后方', '回正，再右转'],
      tip: '转头要慢，感到头晕立即停止', metric: { type: 'count', value: 4 },
      contraindications: ['hypertension'] },
    { id: 'bdt-yaotou', name: '摇头摆尾去心火', category: '简化八段锦', intensity: 'mid', minStanding: 'good',
      desc: '半蹲位，身体左右摇摆，头尾相应，去心火安神',
      steps: ['半蹲，手扶膝', '身体向左摇摆', '身体向右摇摆'],
      tip: '幅度小一些，腰背放松', metric: { type: 'count', value: 4 },
      contraindications: ['hypertension', 'osteoporosis', 'backPain', 'fallHistory'] },
    { id: 'bdt-zanquan', name: '攒拳怒目增气力', category: '简化八段锦', intensity: 'mid', minStanding: 'good',
      desc: '马步，双拳交替向前冲出，怒目而视，增气力',
      steps: ['马步站立', '左拳缓慢冲出', '右拳交替冲出'],
      tip: '马步可高一些，膝盖不超过脚尖', metric: { type: 'count', value: 6 },
      contraindications: ['kneeArthritis', 'kneePain', 'fallHistory'] },
    { id: 'bdt-panzu', name: '双手攀足固肾腰', category: '简化八段锦', intensity: 'mid', minStanding: 'good',
      desc: '站立，双手沿腿缓慢下探至脚面，再缓慢起身',
      steps: ['站立，双臂上举', '弯腰双手探向脚面', '缓慢起身'],
      tip: '弯腰幅度量力而行，起身要慢', metric: { type: 'count', value: 4 },
      contraindications: ['osteoporosis', 'backPain', 'hypertension', 'fallHistory'] }
  ];

  /* ---------- 高危过滤与方案生成 ---------- */
  function isContraindicated(ex, health) {
    for (var i = 0; i < ex.contraindications.length; i++) {
      var c = ex.contraindications[i];
      if (c.indexOf('age:') === 0) {
        if (health.ageGroup === c.slice(4)) return true;
      } else if (health[c]) {
        return true;
      }
    }
    return false;
  }

  function generatePlan(health) {
    health = health || {};
    var standing = health.standing || 'good';
    var rank = standingRank[standing] !== undefined ? standingRank[standing] : 2;

    var safe = exercises.filter(function (ex) {
      if (standingRank[ex.minStanding] > rank) return false;   // 站立能力不足
      if (isContraindicated(ex, health)) return false;          // 命中禁忌症
      return true;
    });

    /* 80岁以上进一步降强度：只保留低强度动作 */
    if (health.ageGroup === '80s') {
      safe = safe.filter(function (ex) { return ex.intensity === 'low'; });
    }

    /* 按类别配额选取，保证方案多样性：坐位操≤3 / 平衡≤3 / 八段锦≤3 */
    var limits = { '坐位操': 3, '平衡训练': 3, '简化八段锦': 3 };
    var picked = [];
    Object.keys(limits).forEach(function (cat) {
      safe.filter(function (ex) { return ex.category === cat; })
          .slice(0, limits[cat])
          .forEach(function (ex) { picked.push(ex); });
    });
    return picked;
  }

  /* ---------- 健康标签文字化 ---------- */
  function describeTags(health) {
    health = health || {};
    var parts = [];
    if (health.ageGroup) parts.push({ '60s': '60-69岁', '70s': '70-79岁', '80s': '80岁以上' }[health.ageGroup] || '');
    if (health.hypertension) parts.push('高血压');
    if (health.kneeArthritis) parts.push('膝关节炎');
    if (health.osteoporosis) parts.push('骨质疏松');
    if (health.diabetes) parts.push('糖尿病');
    if (health.kneePain) parts.push('膝盖疼痛');
    if (health.backPain) parts.push('腰背疼痛');
    if (health.fallHistory) parts.push('有跌倒史');
    if (health.standing) parts.push({ good: '站立良好', assist: '需扶靠站立', poor: '以坐为主' }[health.standing] || '');
    return parts.filter(Boolean);
  }

  /* ---------- 方案页渲染 ---------- */
  function renderPlan() {
    var health = SA.state.health || {};
    var plan = SA.state.plan || [];
    var tagsEl = SA.$('plan-tags');
    var listEl = SA.$('plan-list');
    if (!tagsEl || !listEl) return;

    var tags = describeTags(health);
    tagsEl.textContent = tags.length ? '健康标签：' + tags.join('、') : '';

    listEl.innerHTML = '';
    plan.forEach(function (ex, i) {
      var item = document.createElement('div');
      item.className = 'plan-item';

      var num = document.createElement('div');
      num.className = 'num';
      num.textContent = (i + 1);

      var body = document.createElement('div');
      var h3 = document.createElement('h3');
      h3.textContent = ex.name;
      var p = document.createElement('p');
      p.textContent = ex.desc;
      var tagSpan = document.createElement('span');
      tagSpan.className = 'tag';
      tagSpan.textContent = ex.category;
      body.appendChild(h3);
      body.appendChild(p);
      body.appendChild(tagSpan);

      item.appendChild(num);
      item.appendChild(body);
      listEl.appendChild(item);
    });

    /* 语音播报方案概要 */
    if (SA.voice && SA.voice.enabled && plan.length) {
      var names = plan.map(function (ex) { return ex.name; }).join('、');
      SA.voice.speak('方案给您准备好啦！一共' + plan.length + '个动作，有' + names + '。都是照着您的身体挑的，安全又合适。您看一看，觉得行，就点下面的开始训练。');
    }
  }

  /* ---------- 暴露 ---------- */
  SA.planner = {
    exercises: exercises,
    generatePlan: generatePlan,
    isContraindicated: isContraindicated,
    describeTags: describeTags,
    render: renderPlan
  };

  /* 进入方案屏时自动渲染 */
  var prevOSC = SA.onScreenChange;
  SA.onScreenChange = function (name) {
    if (typeof prevOSC === 'function') prevOSC(name);
    if (name === 'plan') renderPlan();
  };
})();