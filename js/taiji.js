/* 银龄安动 - 24式太极拳课程模块
 * 独立于常规康复方案，在动作选择页（方案屏）单独加载渲染
 * 设计要点：
 *  - 不修改 planner.generatePlan 核心逻辑，不污染 SA.planner.exercises
 *  - 通过 SA.onScreenChange 链式挂载，进入方案屏时渲染太极课程区块
 *  - TAIJI24 数据结构与 planner 动作库一致，直接兼容 train.js 训练流程
 *  - 按站立能力/禁忌症/高龄过滤，高难度招式（蹬脚/下势独立等）对高危老人禁忌
 *  - 节奏舒缓：metric 以 seconds 保持型为主，贴合太极慢动作
 */
(function () {
  'use strict';
  window.SA = window.SA || {};

  var standingRank = { poor: 0, assist: 1, good: 2 };

  /* ---------- 24式太极拳课程 ---------- */
  var TAIJI24 = [
    { id: 'tj-qishi', name: '第一式·起势', category: '24式太极', intensity: 'low', minStanding: 'assist',
      desc: '两脚开立，与肩同宽，双手缓慢上抬至肩高，再缓缓下按，配合呼吸',
      steps: ['两脚开立，与肩同宽', '双臂缓慢上抬至肩高', '屈膝下按至腹前'],
      tip: '沉肩坠肘，呼吸自然，不急不躁', metric: { type: 'seconds', value: 8 },
      contraindications: [] },
    { id: 'tj-yema', name: '第二式·左右野马分鬃', category: '24式太极', intensity: 'mid', minStanding: 'good',
      desc: '弓步分掌，左右交替，如野马分鬃之势，舒展大方',
      steps: ['收脚抱手', '转身上步成弓步', '双手前后分掌', '左右交替'],
      tip: '重心慢慢转移，弓步膝盖不超过脚尖', metric: { type: 'count', value: 2 },
      contraindications: ['fallHistory', 'kneeArthritis'] },
    { id: 'tj-baihe', name: '第三式·白鹤亮翅', category: '24式太极', intensity: 'low', minStanding: 'assist',
      desc: '虚步亮掌，一手上掤一手下按，如白鹤展翅，轻盈安稳',
      steps: ['跟步抱手', '虚步分开，右手上掤', '左手下按于胯旁'],
      tip: '虚步重心在后脚，前脚轻点地面', metric: { type: 'seconds', value: 6 },
      contraindications: ['fallHistory'] },
    { id: 'tj-louxi', name: '第四式·左右搂膝拗步', category: '24式太极', intensity: 'mid', minStanding: 'good',
      desc: '弓步搂膝推掌，一手搂过膝前，一手向前推出，左右交替',
      steps: ['转腰收脚', '上步弓腿', '一手搂膝，一手推掌'],
      tip: '转腰要缓，推掌柔和，呼吸顺畅', metric: { type: 'count', value: 2 },
      contraindications: ['fallHistory', 'kneeArthritis'] },
    { id: 'tj-pipa', name: '第五式·手挥琵琶', category: '24式太极', intensity: 'low', minStanding: 'assist',
      desc: '虚步合手，如怀抱琵琶，前后手相应，安舒中正',
      steps: ['跟步后坐', '虚步合手，掌心相对'],
      tip: '立身中正，松腰敛臀', metric: { type: 'seconds', value: 6 },
      contraindications: ['fallHistory'] },
    { id: 'tj-daojuan', name: '第六式·左右倒卷肱', category: '24式太极', intensity: 'mid', minStanding: 'good',
      desc: '退步推掌，一手后撤一手前推，如倒卷珠帘，左右交替',
      steps: ['转腰撤手', '退步虚步', '前手推掌后手撤'],
      tip: '退步要稳，先落脚再移重心', metric: { type: 'count', value: 2 },
      contraindications: ['fallHistory'] },
    { id: 'tj-lanwei-zuo', name: '第七式·左揽雀尾', category: '24式太极', intensity: 'mid', minStanding: 'good',
      desc: '掤、捋、挤、按四手连贯，如揽雀尾，圆活连贯',
      steps: ['弓步掤手', '坐身捋回', '转身前挤', '弓步前按'],
      tip: '掤捋挤按一气呵成，劲力柔和', metric: { type: 'seconds', value: 10 },
      contraindications: ['fallHistory', 'kneeArthritis'] },
    { id: 'tj-lanwei-you', name: '第八式·右揽雀尾', category: '24式太极', intensity: 'mid', minStanding: 'good',
      desc: '转身右做揽雀尾，掤捋挤按，与左式对称呼应',
      steps: ['扣脚转身', '弓步掤手', '坐身捋回', '前挤前按'],
      tip: '转身要慢，脚下站稳再动手', metric: { type: 'seconds', value: 10 },
      contraindications: ['fallHistory', 'kneeArthritis'] },
    { id: 'tj-danbian', name: '第九式·单鞭', category: '24式太极', intensity: 'low', minStanding: 'good',
      desc: '弓步勾手推掌，一手勾挂一手单鞭前推，舒展大方',
      steps: ['扣脚勾手', '转身弓步', '单掌前推'],
      tip: '推掌如抽鞭，柔中带刚', metric: { type: 'seconds', value: 6 },
      contraindications: ['fallHistory'] },
    { id: 'tj-yunshou', name: '第十式·云手', category: '24式太极', intensity: 'low', minStanding: 'good',
      desc: '横移步云手，双手交替在体前划圆，如云行空中',
      steps: ['扣脚转腰', '左右云手划圆', '横移脚步配合'],
      tip: '眼随手走，脚下如行云流水', metric: { type: 'count', value: 3 },
      contraindications: ['fallHistory'] },
    { id: 'tj-danbian-2', name: '第十一式·单鞭', category: '24式太极', intensity: 'low', minStanding: 'good',
      desc: '云手之后再次单鞭，勾手推掌，与第九式呼应',
      steps: ['扣脚勾手', '转身弓步', '单掌前推'],
      tip: '动作与前面单鞭相同，慢慢来', metric: { type: 'seconds', value: 6 },
      contraindications: ['fallHistory'] },
    { id: 'tj-gaotanma', name: '第十二式·高探马', category: '24式太极', intensity: 'low', minStanding: 'assist',
      desc: '虚步探掌，后脚跟进，一手回带一手前探，轻灵安稳',
      steps: ['跟步后坐', '虚步前探掌'],
      tip: '探掌高度在肩前，不要耸肩', metric: { type: 'seconds', value: 6 },
      contraindications: ['fallHistory'] },
    { id: 'tj-dengjiao-you', name: '第十三式·右蹬脚', category: '24式太极', intensity: 'mid', minStanding: 'good',
      desc: '单腿支撑，右脚缓慢蹬出，双手十字分开，稳而缓',
      steps: ['穿掌十字手', '提膝蹬脚', '双手分开'],
      tip: '蹬脚高度量力而行，扶稳再抬腿', metric: { type: 'seconds', value: 4 },
      contraindications: ['fallHistory', 'osteoporosis'] },
    { id: 'tj-shuangfeng', name: '第十四式·双峰贯耳', category: '24式太极', intensity: 'mid', minStanding: 'good',
      desc: '弓步双拳贯击，两拳从两侧向前合击，如双峰贯耳',
      steps: ['收脚落手', '弓步双拳前贯'],
      tip: '贯拳柔和，肘部微屈不僵直', metric: { type: 'seconds', value: 6 },
      contraindications: ['fallHistory', 'kneeArthritis'] },
    { id: 'tj-dengjiao-zuo', name: '第十五式·转身左蹬脚', category: '24式太极', intensity: 'mid', minStanding: 'good',
      desc: '转身后左脚蹬出，与右蹬脚对称，转身要稳',
      steps: ['转身十字手', '提膝蹬脚', '双手分开'],
      tip: '转身慢一点，站稳后再蹬脚', metric: { type: 'seconds', value: 4 },
      contraindications: ['fallHistory', 'osteoporosis'] },
    { id: 'tj-xiashi-zuo', name: '第十六式·左下势独立', category: '24式太极', intensity: 'mid', minStanding: 'good',
      desc: '仆步下势后起身独立，一手挑掌一手按胯，起落缓慢',
      steps: ['收脚提手', '仆步下势', '起身独立挑掌'],
      tip: '仆步幅度量力而行，起身要慢', metric: { type: 'seconds', value: 8 },
      contraindications: ['fallHistory', 'osteoporosis', 'kneeArthritis', 'kneePain'] },
    { id: 'tj-xiashi-you', name: '第十七式·右下势独立', category: '24式太极', intensity: 'mid', minStanding: 'good',
      desc: '右仆步下势后起身独立，与左式对称呼应',
      steps: ['落脚转腰', '仆步下势', '起身独立挑掌'],
      tip: '独立站稳再挑掌，不急不慌', metric: { type: 'seconds', value: 8 },
      contraindications: ['fallHistory', 'osteoporosis', 'kneeArthritis', 'kneePain'] },
    { id: 'tj-chuansuo', name: '第十八式·左右穿梭', category: '24式太极', intensity: 'mid', minStanding: 'good',
      desc: '弓步架推，一手架于头上一手前推，左右穿梭如织',
      steps: ['落脚转腰', '弓步架推掌', '左右交替'],
      tip: '架推协调，转身连贯圆活', metric: { type: 'count', value: 2 },
      contraindications: ['fallHistory', 'kneeArthritis'] },
    { id: 'tj-haidizhen', name: '第十九式·海底针', category: '24式太极', intensity: 'mid', minStanding: 'good',
      desc: '虚步下探，右手如针探海，向下插掌，腰脊松沉',
      steps: ['跟步后坐', '虚步下插掌'],
      tip: '弯腰幅度量力而行，起身要慢', metric: { type: 'seconds', value: 6 },
      contraindications: ['osteoporosis', 'backPain', 'hypertension'] },
    { id: 'tj-shantongbi', name: '第二十式·闪通臂', category: '24式太极', intensity: 'mid', minStanding: 'good',
      desc: '弓步架推，一手架一手推，如闪通臂膀，舒展有力',
      steps: ['提手收脚', '弓步架推'],
      tip: '架推同时到位，腰背舒展', metric: { type: 'seconds', value: 6 },
      contraindications: ['fallHistory', 'backPain'] },
    { id: 'tj-banlan', name: '第二十一式·转身搬拦捶', category: '24式太极', intensity: 'mid', minStanding: 'good',
      desc: '转身后搬、拦、捶三手连贯，弓步冲拳，刚柔相济',
      steps: ['转身搬手', '上步拦手', '弓步冲拳'],
      tip: '搬拦捶一气呵成，冲拳不用猛力', metric: { type: 'seconds', value: 8 },
      contraindications: ['fallHistory', 'kneeArthritis'] },
    { id: 'tj-rufeng', name: '第二十二式·如封似闭', category: '24式太极', intensity: 'low', minStanding: 'good',
      desc: '弓步双按，双手回收再向前按出，如封似闭，含蓄沉稳',
      steps: ['收手后坐', '弓步双按掌'],
      tip: '按掌柔和，呼吸自然下沉', metric: { type: 'seconds', value: 6 },
      contraindications: ['fallHistory'] },
    { id: 'tj-shizishou', name: '第二十三式·十字手', category: '24式太极', intensity: 'low', minStanding: 'assist',
      desc: '开立步合抱，双手在胸前交叉成十字，安舒收束',
      steps: ['扣脚开步', '双手合抱交叉'],
      tip: '合抱圆满，松静自然', metric: { type: 'seconds', value: 6 },
      contraindications: [] },
    { id: 'tj-shoushi', name: '第二十四式·收势', category: '24式太极', intensity: 'low', minStanding: 'assist',
      desc: '双手缓缓下按至胯旁，收脚并步，气息归元，课程圆满',
      steps: ['双臂下按', '收脚并步', '静立调息'],
      tip: '气息慢慢平复，不急着走动', metric: { type: 'seconds', value: 8 },
      contraindications: [] }
  ];

  /* ---------- 高危过滤（与 planner 同规则，独立实现避免耦合） ---------- */
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

  /* 按健康标签过滤安全招式：站立能力 → 禁忌症 → 80岁以上仅低强度 */
  function getSafeCourse(health) {
    health = health || {};
    var standing = health.standing || 'good';
    var rank = standingRank[standing] !== undefined ? standingRank[standing] : 2;
    var safe = TAIJI24.filter(function (ex) {
      if (standingRank[ex.minStanding] > rank) return false;
      if (isContraindicated(ex, health)) return false;
      return true;
    });
    if (health.ageGroup === '80s') {
      safe = safe.filter(function (ex) { return ex.intensity === 'low'; });
    }
    return safe;
  }

  /* ---------- 方案页（动作选择页）太极课程区块渲染 ---------- */
  function renderCourse() {
    var sec = SA.$ && SA.$('taiji-course-section');
    if (!sec) return;                       /* 容器不存在（如测试页）静默跳过 */
    var health = (SA.state && SA.state.health) || {};
    var course = getSafeCourse(health);
    var listEl = SA.$('taiji-course-list');
    var countEl = SA.$('taiji-course-count');
    var btn = SA.$('btn-start-taiji');

    if (!course.length) {                   /* 无安全招式：隐藏整个区块 */
      sec.hidden = true;
      return;
    }
    sec.hidden = false;
    if (countEl) countEl.textContent = String(course.length);
    if (listEl) {
      listEl.innerHTML = '';
      course.forEach(function (ex, i) {
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
    }
    if (btn) btn.disabled = false;
  }

  /* ---------- 启动太极课程：替换方案为 TAIJI24 安全子集后进入训练 ---------- */
  function startCourse() {
    var health = (SA.state && SA.state.health) || {};
    var course = getSafeCourse(health);
    if (!course.length) {
      if (SA.voice && SA.voice.enabled) {
        SA.voice.speak('这套太极课程不太适合现在的身体情况，咱们先练康复方案，一样很棒。');
      }
      return;
    }
    SA.state.plan = course;
    SA.state.trainIndex = 0;
    if (typeof SA.startTraining === 'function') SA.startTraining();
  }

  /* ---------- 暴露 ---------- */
  SA.taiji = {
    TAIJI24: TAIJI24,
    getSafeCourse: getSafeCourse,
    isContraindicated: isContraindicated,
    renderCourse: renderCourse,
    startCourse: startCourse
  };

  /* 链式挂载：进入方案屏时渲染太极课程区块（不干扰 planner.renderPlan） */
  var prevOSC = SA.onScreenChange;
  SA.onScreenChange = function (name) {
    if (typeof prevOSC === 'function') prevOSC(name);
    if (name === 'plan') renderCourse();
  };

  document.addEventListener('DOMContentLoaded', function () {
    if (!SA.$) return;
    var btn = SA.$('btn-start-taiji');
    if (btn) btn.addEventListener('click', startCourse);
  });

  console.log('[太极模块] TAIJI24 已加载，共 ' + TAIJI24.length + ' 式');
})();