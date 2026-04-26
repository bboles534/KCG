const canvas = document.querySelector("#game");
const ctx = canvas.getContext("2d");

const overlay = document.querySelector("#overlay");
const overlayTitle = document.querySelector("#overlay-title");
const overlayText = document.querySelector("#overlay-text");
const startButton = document.querySelector("#start-button");
const dashButton = document.querySelector("#dash-button");
const overdriveButton = document.querySelector("#overdrive-button");
const scoreEl = document.querySelector("#score");
const waveEl = document.querySelector("#wave");
const comboEl = document.querySelector("#combo");
const bestEl = document.querySelector("#best");
const overdriveEl = document.querySelector("#overdrive");

const DPR = Math.min(window.devicePixelRatio || 1, 2);
const BEST_SCORE_KEY = "neon-rift-best-score";

const state = {
  running: false,
  score: 0,
  best: Number(localStorage.getItem(BEST_SCORE_KEY) || 0),
  wave: 1,
  combo: 1,
  comboTimer: 0,
  overdrive: 0,
  overdriveActive: 0,
  waveTimer: 0,
  spawnBudget: 0,
  time: 0,
  lastFrame: 0,
  screenShake: 0,
  pointer: {
    x: 0,
    y: 0,
    active: false
  },
  keys: new Set(),
  touchMove: false,
  entities: {
    bullets: [],
    enemies: [],
    particles: [],
    shards: []
  },
  player: null
};

resize();
reset();
updateHud();
bestEl.textContent = state.best.toLocaleString();

window.addEventListener("resize", resize);
window.addEventListener("keydown", onKeyDown);
window.addEventListener("keyup", (event) => state.keys.delete(event.key.toLowerCase()));
canvas.addEventListener("pointerdown", onPointerDown);
canvas.addEventListener("pointermove", onPointerMove);
window.addEventListener("pointerup", onPointerUp);
startButton.addEventListener("click", startGame);
dashButton.addEventListener("click", () => {
  if (state.running) {
    triggerDash();
  }
});
overdriveButton.addEventListener("click", () => {
  if (state.running) {
    triggerOverdrive();
  }
});

requestAnimationFrame(loop);

function startGame() {
  reset();
  state.running = true;
  overlay.classList.remove("visible");
}

function reset() {
  state.running = false;
  state.score = 0;
  state.wave = 1;
  state.combo = 1;
  state.comboTimer = 0;
  state.overdrive = 0;
  state.overdriveActive = 0;
  state.waveTimer = 0;
  state.spawnBudget = 10;
  state.time = 0;
  state.screenShake = 0;
  state.entities.bullets = [];
  state.entities.enemies = [];
  state.entities.particles = [];
  state.entities.shards = [];
  overlayTitle.textContent = "Enter the Rift";
  overlayText.textContent =
    "Dodge swarms, keep the combo alive, and hit overdrive at the right moment. The arena scales forever.";
  state.player = {
    x: canvas.width / 2,
    y: canvas.height / 2,
    radius: 18,
    speed: 380,
    hp: 100,
    maxHp: 100,
    angle: 0,
    cooldown: 0,
    dashCooldown: 0,
    dashTimer: 0,
    invulnerable: 0
  };
}

function resize() {
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * DPR);
  canvas.height = Math.round(rect.height * DPR);
  if (state.player) {
    state.player.x = clamp(state.player.x, 0, canvas.width);
    state.player.y = clamp(state.player.y, 0, canvas.height);
  }
}

function onKeyDown(event) {
  const key = event.key.toLowerCase();
  if ([" ", "arrowup", "arrowdown", "arrowleft", "arrowright", "shift"].includes(key)) {
    event.preventDefault();
  }

  if (!state.running && (key === "enter" || key === " ")) {
    startGame();
    return;
  }

  if (state.running && key === "shift") {
    triggerDash();
  }

  if (state.running && key === "e") {
    triggerOverdrive();
  }

  state.keys.add(key);
}

function onPointerDown(event) {
  const point = canvasPoint(event);
  state.pointer.x = point.x;
  state.pointer.y = point.y;
  state.pointer.active = true;
  state.touchMove = event.pointerType !== "mouse";

  if (!state.running) {
    startGame();
  }
}

function onPointerMove(event) {
  const point = canvasPoint(event);
  state.pointer.x = point.x;
  state.pointer.y = point.y;
}

function onPointerUp() {
  state.pointer.active = false;
  state.touchMove = false;
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * DPR,
    y: (event.clientY - rect.top) * DPR
  };
}

function loop(timestamp) {
  const dt = Math.min((timestamp - (state.lastFrame || timestamp)) / 1000, 0.033);
  state.lastFrame = timestamp;

  if (state.running) {
    update(dt);
  }
  render();
  requestAnimationFrame(loop);
}

function update(dt) {
  state.time += dt;
  state.waveTimer += dt;
  state.screenShake = Math.max(0, state.screenShake - dt * 5);

  if (state.comboTimer > 0) {
    state.comboTimer -= dt;
  } else {
    state.combo = approach(state.combo, 1, dt * 2.2);
  }

  if (state.overdriveActive > 0) {
    state.overdriveActive -= dt;
  }

  updatePlayer(dt);
  updateSpawning(dt);
  updateBullets(dt);
  updateEnemies(dt);
  updateShards(dt);
  updateParticles(dt);
  updateHud();

  if (state.player.hp <= 0) {
    endGame();
  }
}

function updatePlayer(dt) {
  const player = state.player;
  player.cooldown = Math.max(0, player.cooldown - dt);
  player.dashCooldown = Math.max(0, player.dashCooldown - dt);
  player.dashTimer = Math.max(0, player.dashTimer - dt);
  player.invulnerable = Math.max(0, player.invulnerable - dt);

  let moveX = 0;
  let moveY = 0;

  if (state.keys.has("w") || state.keys.has("arrowup")) {
    moveY -= 1;
  }
  if (state.keys.has("s") || state.keys.has("arrowdown")) {
    moveY += 1;
  }
  if (state.keys.has("a") || state.keys.has("arrowleft")) {
    moveX -= 1;
  }
  if (state.keys.has("d") || state.keys.has("arrowright")) {
    moveX += 1;
  }

  if (state.touchMove && state.pointer.active) {
    moveX += (state.pointer.x - player.x) / 120;
    moveY += (state.pointer.y - player.y) / 120;
  }

  const magnitude = Math.hypot(moveX, moveY);
  if (magnitude > 0) {
    const speedBoost = player.dashTimer > 0 ? 2.3 : 1;
    player.x += (moveX / magnitude) * player.speed * speedBoost * dt;
    player.y += (moveY / magnitude) * player.speed * speedBoost * dt;
  }

  player.x = clamp(player.x, player.radius, canvas.width - player.radius);
  player.y = clamp(player.y, player.radius, canvas.height - player.radius);

  const target = acquireAimTarget();
  player.angle = Math.atan2(target.y - player.y, target.x - player.x);

  const wantsFire = state.pointer.active || state.keys.has(" ");
  if (wantsFire && player.cooldown <= 0) {
    fireBullet(player);
  }
}

function acquireAimTarget() {
  if (state.pointer.active || state.pointer.x || state.pointer.y) {
    return state.pointer;
  }

  let nearest = null;
  let nearestDistance = Infinity;
  for (const enemy of state.entities.enemies) {
    const currentDistance = Math.hypot(enemy.x - state.player.x, enemy.y - state.player.y);
    if (currentDistance < nearestDistance) {
      nearestDistance = currentDistance;
      nearest = enemy;
    }
  }

  return nearest ?? { x: canvas.width / 2, y: canvas.height / 2 };
}

function triggerDash() {
  const player = state.player;
  if (player.dashCooldown > 0) {
    return;
  }

  player.dashCooldown = 1.8;
  player.dashTimer = 0.16;
  player.invulnerable = 0.18;
  state.screenShake = 0.4;

  for (let i = 0; i < 12; i += 1) {
    spawnParticle(player.x, player.y, {
      speed: random(120, 320),
      color: "#54f3ff",
      size: random(2, 5),
      life: random(0.18, 0.4)
    });
  }
}

function triggerOverdrive() {
  if (state.overdrive < 100 || state.overdriveActive > 0) {
    return;
  }

  state.overdrive = 0;
  state.overdriveActive = 8;
  state.screenShake = 0.6;

  for (let i = 0; i < 30; i += 1) {
    spawnParticle(state.player.x, state.player.y, {
      speed: random(160, 480),
      color: i % 2 === 0 ? "#ffc857" : "#ff4fd8",
      size: random(3, 7),
      life: random(0.35, 0.9)
    });
  }
}

function fireBullet(player) {
  const fireRate = state.overdriveActive > 0 ? 0.065 : 0.16;
  player.cooldown = fireRate;

  const spread = state.overdriveActive > 0 ? 0.16 : 0.07;
  const count = state.overdriveActive > 0 ? 3 : 1;

  for (let i = 0; i < count; i += 1) {
    const offset = count === 1 ? 0 : mapRange(i, 0, count - 1, -spread, spread);
    const angle = player.angle + offset;
    state.entities.bullets.push({
      x: player.x + Math.cos(angle) * 22,
      y: player.y + Math.sin(angle) * 22,
      vx: Math.cos(angle) * (state.overdriveActive > 0 ? 960 : 760),
      vy: Math.sin(angle) * (state.overdriveActive > 0 ? 960 : 760),
      radius: state.overdriveActive > 0 ? 6 : 4,
      damage: state.overdriveActive > 0 ? 34 : 18,
      life: 1.05,
      pierce: state.overdriveActive > 0 ? 3 : 0,
      hostile: false,
      color: state.overdriveActive > 0 ? "#ffc857" : "#54f3ff"
    });
  }
}

function updateSpawning(dt) {
  const targetWaveTime = Math.max(7, 14 - state.wave * 0.25);
  if (state.waveTimer >= targetWaveTime) {
    state.wave += 1;
    state.waveTimer = 0;
    state.spawnBudget += 8 + state.wave * 2;
    if (state.wave % 5 === 0) {
      spawnBoss();
    }
  }

  const activeCost = state.entities.enemies.reduce((sum, enemy) => sum + enemy.cost, 0);
  if (activeCost >= 18 + state.wave * 2 || state.spawnBudget <= 0) {
    return;
  }

  const chance = dt * Math.min(5.5, 1.2 + state.wave * 0.16);
  if (Math.random() < chance) {
    spawnEnemy();
  }
}

function spawnEnemy() {
  const edge = Math.floor(Math.random() * 4);
  const margin = 50;
  let x = 0;
  let y = 0;

  if (edge === 0) {
    x = random(0, canvas.width);
    y = -margin;
  } else if (edge === 1) {
    x = canvas.width + margin;
    y = random(0, canvas.height);
  } else if (edge === 2) {
    x = random(0, canvas.width);
    y = canvas.height + margin;
  } else {
    x = -margin;
    y = random(0, canvas.height);
  }

  const roll = Math.random();
  let enemy;

  if (roll < 0.16 + state.wave * 0.005) {
    enemy = {
      kind: "spinner",
      x,
      y,
      radius: 16,
      speed: random(120, 170) + state.wave * 4,
      hp: 32 + state.wave * 7,
      maxHp: 32 + state.wave * 7,
      value: 90,
      cost: 3,
      orbit: random(1.8, 3.4),
      pulse: 0,
      color: "#ff4fd8"
    };
  } else if (roll < 0.34) {
    enemy = {
      kind: "brute",
      x,
      y,
      radius: 25,
      speed: random(70, 95) + state.wave * 2.4,
      hp: 84 + state.wave * 14,
      maxHp: 84 + state.wave * 14,
      value: 150,
      cost: 5,
      pulse: 0,
      color: "#ffc857"
    };
  } else {
    enemy = {
      kind: "chaser",
      x,
      y,
      radius: 14,
      speed: random(100, 150) + state.wave * 3.2,
      hp: 24 + state.wave * 6,
      maxHp: 24 + state.wave * 6,
      value: 70,
      cost: 2,
      pulse: 0,
      color: "#8fff9f"
    };
  }

  state.entities.enemies.push(enemy);
  state.spawnBudget -= enemy.cost;
}

function spawnBoss() {
  state.entities.enemies.push({
    kind: "boss",
    x: canvas.width / 2,
    y: -120,
    radius: 60,
    speed: 54 + state.wave * 1.4,
    hp: 820 + state.wave * 90,
    maxHp: 820 + state.wave * 90,
    value: 1600,
    cost: 18,
    phase: 0,
    pulse: 0,
    color: "#ff7b54"
  });
}

function updateBullets(dt) {
  const enemiesToDestroy = new Set();

  state.entities.bullets = state.entities.bullets.filter((bullet) => {
    bullet.x += bullet.vx * dt;
    bullet.y += bullet.vy * dt;
    bullet.life -= dt;

    let alive = bullet.life > 0 && inBounds(bullet.x, bullet.y, 80);
    if (!alive) {
      return false;
    }

    if (bullet.hostile) {
      return true;
    }

    for (const enemy of state.entities.enemies) {
      if (enemiesToDestroy.has(enemy)) {
        continue;
      }

      if (distance(bullet, enemy) <= bullet.radius + enemy.radius) {
        enemy.hp -= bullet.damage;
        alive = bullet.pierce > 0;
        bullet.pierce -= 1;
        bullet.damage *= 0.88;

        burst(enemy.x, enemy.y, enemy.color, 5);
        if (enemy.hp <= 0) {
          enemiesToDestroy.add(enemy);
        }
        if (!alive) {
          break;
        }
      }
    }

    return alive;
  });

  for (const enemy of enemiesToDestroy) {
    destroyEnemy(enemy);
  }
}

function updateEnemies(dt) {
  const player = state.player;
  const remaining = [];
  const enemiesToDestroy = new Set();

  for (const enemy of state.entities.enemies) {
    enemy.pulse = (enemy.pulse || 0) + dt;

    if (enemy.kind === "spinner") {
      const angle = Math.atan2(player.y - enemy.y, player.x - enemy.x);
      enemy.x += Math.cos(angle + Math.sin(state.time * enemy.orbit) * 0.8) * enemy.speed * dt;
      enemy.y += Math.sin(angle + Math.sin(state.time * enemy.orbit) * 0.8) * enemy.speed * dt;
    } else if (enemy.kind === "boss") {
      const angle = Math.atan2(player.y - enemy.y, player.x - enemy.x);
      enemy.phase += dt;
      enemy.x += Math.cos(angle + Math.sin(enemy.phase * 2) * 0.45) * enemy.speed * dt;
      enemy.y += Math.sin(angle) * enemy.speed * dt;

      if (Math.random() < dt * 3.4) {
        const shardAngle = random(0, Math.PI * 2);
        state.entities.bullets.push({
          x: enemy.x + Math.cos(shardAngle) * 26,
          y: enemy.y + Math.sin(shardAngle) * 26,
          vx: Math.cos(shardAngle) * 280,
          vy: Math.sin(shardAngle) * 280,
          radius: 5,
          damage: 16,
          life: 2.4,
          hostile: true,
          pierce: 0,
          color: "#ff7b54"
        });
      }
    } else {
      const angle = Math.atan2(player.y - enemy.y, player.x - enemy.x);
      enemy.x += Math.cos(angle) * enemy.speed * dt;
      enemy.y += Math.sin(angle) * enemy.speed * dt;
    }

    if (distance(enemy, player) <= enemy.radius + player.radius) {
      if (player.invulnerable <= 0) {
        player.hp -= enemy.kind === "boss" ? 30 : enemy.kind === "brute" ? 20 : 12;
        player.invulnerable = 0.45;
        state.screenShake = 0.55;
        burst(player.x, player.y, "#ffffff", 14);
      }

      if (enemy.kind !== "boss") {
        enemiesToDestroy.add(enemy);
      }
    }

    if (enemy.hp > 0 && !enemiesToDestroy.has(enemy)) {
      remaining.push(enemy);
    }
  }

  state.entities.enemies = remaining;

  for (const enemy of enemiesToDestroy) {
    if (enemy.kind !== "boss") {
      burst(enemy.x, enemy.y, enemy.color, 12);
    }
  }

  state.entities.bullets = state.entities.bullets.filter((bullet) => {
    if (!bullet.hostile) {
      return true;
    }

    if (distance(bullet, player) <= bullet.radius + player.radius) {
      if (player.invulnerable <= 0) {
        player.hp -= bullet.damage;
        player.invulnerable = 0.35;
        state.screenShake = 0.42;
      }
      return false;
    }

    return bullet.life > 0 && inBounds(bullet.x, bullet.y, 80);
  });
}

function updateShards(dt) {
  const player = state.player;
  state.entities.shards = state.entities.shards.filter((shard) => {
    shard.life -= dt;
    const angle = Math.atan2(player.y - shard.y, player.x - shard.x);
    shard.vx += Math.cos(angle) * 24 * dt;
    shard.vy += Math.sin(angle) * 24 * dt;
    shard.x += shard.vx * dt;
    shard.y += shard.vy * dt;

    if (distance(shard, player) <= shard.radius + player.radius + 8) {
      if (shard.kind === "heal") {
        player.hp = Math.min(player.maxHp, player.hp + 10);
        state.overdrive = Math.min(100, state.overdrive + 12);
      } else {
        state.combo = Math.min(6, state.combo + 0.25);
        state.comboTimer = 4;
        state.score += 40;
      }
      burst(shard.x, shard.y, shard.color, 8);
      return false;
    }

    return shard.life > 0;
  });
}

function updateParticles(dt) {
  state.entities.particles = state.entities.particles.filter((particle) => {
    particle.life -= dt;
    particle.x += particle.vx * dt;
    particle.y += particle.vy * dt;
    particle.vx *= 0.98;
    particle.vy *= 0.98;
    return particle.life > 0;
  });
}

function destroyEnemy(enemy) {
  const index = state.entities.enemies.indexOf(enemy);
  if (index >= 0) {
    state.entities.enemies.splice(index, 1);
  }

  const scoreGain = Math.round(enemy.value * state.combo);
  state.score += scoreGain;
  state.combo = Math.min(8, state.combo + (enemy.kind === "boss" ? 0.8 : 0.12));
  state.comboTimer = 4;
  state.overdrive = Math.min(100, state.overdrive + (enemy.kind === "boss" ? 32 : 6));
  state.screenShake = Math.max(state.screenShake, enemy.kind === "boss" ? 0.9 : 0.24);

  burst(enemy.x, enemy.y, enemy.color, enemy.kind === "boss" ? 42 : 14);
  dropShard(enemy);
}

function dropShard(enemy) {
  const total = enemy.kind === "boss" ? 8 : Math.random() < 0.24 ? 1 : 0;
  for (let i = 0; i < total; i += 1) {
    const heal = enemy.kind === "boss" ? i % 3 === 0 : Math.random() < 0.45;
    state.entities.shards.push({
      x: enemy.x + random(-12, 12),
      y: enemy.y + random(-12, 12),
      vx: random(-90, 90),
      vy: random(-90, 90),
      radius: heal ? 7 : 6,
      life: 9,
      kind: heal ? "heal" : "combo",
      color: heal ? "#54f3ff" : "#ffc857"
    });
  }
}

function endGame() {
  state.running = false;
  state.best = Math.max(state.best, Math.round(state.score));
  localStorage.setItem(BEST_SCORE_KEY, String(state.best));
  bestEl.textContent = state.best.toLocaleString();
  overlay.classList.add("visible");
  overlayTitle.textContent = "Run Over";
  overlayText.textContent =
    `Final score ${Math.round(state.score).toLocaleString()} on wave ${state.wave}. Press Play Now to jump back in.`;
}

function render() {
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.restore();

  const shakeX = random(-1, 1) * state.screenShake * 12;
  const shakeY = random(-1, 1) * state.screenShake * 12;

  ctx.save();
  ctx.translate(shakeX, shakeY);

  drawBackground();
  drawShards();
  drawParticles();
  drawBullets();
  drawEnemies();
  drawPlayer();
  drawArenaFx();

  ctx.restore();
}

function drawBackground() {
  const gradient = ctx.createRadialGradient(
    canvas.width / 2,
    canvas.height / 2,
    canvas.width * 0.05,
    canvas.width / 2,
    canvas.height / 2,
    canvas.width * 0.65
  );
  gradient.addColorStop(0, "rgba(22, 34, 72, 0.35)");
  gradient.addColorStop(1, "rgba(3, 5, 15, 0.9)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  for (let i = 0; i < 80; i += 1) {
    const x = ((i * 183.4 + state.time * 35) % (canvas.width + 160)) - 80;
    const y = (i * 97.3) % canvas.height;
    ctx.fillStyle = i % 3 === 0 ? "rgba(84,243,255,0.16)" : "rgba(255,255,255,0.08)";
    ctx.fillRect(x, y, 2, 2);
  }
}

function drawPlayer() {
  const player = state.player;
  ctx.save();
  ctx.translate(player.x, player.y);
  ctx.rotate(player.angle);

  if (state.overdriveActive > 0) {
    ctx.shadowBlur = 34;
    ctx.shadowColor = "#ffc857";
  } else {
    ctx.shadowBlur = 20;
    ctx.shadowColor = "#54f3ff";
  }

  ctx.fillStyle = player.invulnerable > 0 ? "rgba(255,255,255,0.9)" : "#54f3ff";
  ctx.beginPath();
  ctx.moveTo(24, 0);
  ctx.lineTo(-16, -13);
  ctx.lineTo(-8, 0);
  ctx.lineTo(-16, 13);
  ctx.closePath();
  ctx.fill();

  ctx.fillStyle = "#ff4fd8";
  ctx.fillRect(-18, -4, 8, 8);
  ctx.restore();

  const hpWidth = 170;
  const x = 24;
  const y = canvas.height - 30;
  ctx.fillStyle = "rgba(255,255,255,0.08)";
  ctx.fillRect(x, y, hpWidth, 10);
  ctx.fillStyle = "#54f3ff";
  ctx.fillRect(x, y, hpWidth * (player.hp / player.maxHp), 10);
}

function drawBullets() {
  for (const bullet of state.entities.bullets) {
    ctx.fillStyle = bullet.color;
    ctx.shadowBlur = 20;
    ctx.shadowColor = bullet.color;
    ctx.beginPath();
    ctx.arc(bullet.x, bullet.y, bullet.radius, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.shadowBlur = 0;
}

function drawEnemies() {
  for (const enemy of state.entities.enemies) {
    ctx.save();
    ctx.translate(enemy.x, enemy.y);
    ctx.fillStyle = enemy.color;
    ctx.shadowBlur = enemy.kind === "boss" ? 38 : 16;
    ctx.shadowColor = enemy.color;

    if (enemy.kind === "spinner") {
      ctx.rotate(state.time * 4);
      for (let i = 0; i < 3; i += 1) {
        ctx.rotate((Math.PI * 2) / 3);
        ctx.fillRect(enemy.radius * 0.2, -4, enemy.radius * 1.2, 8);
      }
      ctx.beginPath();
      ctx.arc(0, 0, enemy.radius * 0.7, 0, Math.PI * 2);
      ctx.fill();
    } else if (enemy.kind === "brute") {
      roundedRect(-enemy.radius, -enemy.radius, enemy.radius * 2, enemy.radius * 2, 9);
      ctx.fill();
    } else if (enemy.kind === "boss") {
      ctx.rotate(Math.sin(enemy.pulse * 1.5) * 0.15);
      for (let i = 0; i < 6; i += 1) {
        ctx.rotate(Math.PI / 3);
        ctx.fillRect(enemy.radius * 0.3, -6, enemy.radius * 1.25, 12);
      }
      ctx.beginPath();
      ctx.arc(0, 0, enemy.radius * 0.85, 0, Math.PI * 2);
      ctx.fill();
    } else {
      ctx.beginPath();
      ctx.arc(0, 0, enemy.radius, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();

    const hpRatio = enemy.hp / enemy.maxHp;
    const barWidth = enemy.radius * 2.2;
    ctx.fillStyle = "rgba(255,255,255,0.08)";
    ctx.fillRect(enemy.x - barWidth / 2, enemy.y + enemy.radius + 10, barWidth, 5);
    ctx.fillStyle = enemy.color;
    ctx.fillRect(enemy.x - barWidth / 2, enemy.y + enemy.radius + 10, barWidth * hpRatio, 5);
  }

  ctx.shadowBlur = 0;
}

function drawParticles() {
  for (const particle of state.entities.particles) {
    ctx.globalAlpha = Math.max(0, particle.life / particle.maxLife);
    ctx.fillStyle = particle.color;
    ctx.beginPath();
    ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

function drawShards() {
  for (const shard of state.entities.shards) {
    ctx.save();
    ctx.translate(shard.x, shard.y);
    ctx.rotate(state.time * 3.5);
    ctx.fillStyle = shard.color;
    ctx.shadowBlur = 18;
    ctx.shadowColor = shard.color;
    ctx.beginPath();
    ctx.moveTo(0, -shard.radius);
    ctx.lineTo(shard.radius, 0);
    ctx.lineTo(0, shard.radius);
    ctx.lineTo(-shard.radius, 0);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }
  ctx.shadowBlur = 0;
}

function drawArenaFx() {
  ctx.strokeStyle = "rgba(84,243,255,0.08)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(canvas.width / 2, canvas.height / 2, Math.min(canvas.width, canvas.height) * 0.34, 0, Math.PI * 2);
  ctx.stroke();

  if (state.overdriveActive > 0) {
    ctx.strokeStyle = "rgba(255,200,87,0.35)";
    ctx.lineWidth = 6;
    ctx.strokeRect(10, 10, canvas.width - 20, canvas.height - 20);
  }
}

function updateHud() {
  scoreEl.textContent = Math.round(state.score).toLocaleString();
  waveEl.textContent = state.wave.toString();
  comboEl.textContent = `x${state.combo.toFixed(1)}`;
  overdriveEl.textContent = `${Math.round(state.overdrive)}%`;
  dashButton.textContent = state.player.dashCooldown > 0 ? `Dash ${state.player.dashCooldown.toFixed(1)}s` : "Dash";
  overdriveButton.textContent = state.overdrive >= 100 ? "Overdrive Ready" : `Overdrive ${Math.round(state.overdrive)}%`;
}

function burst(x, y, color, count) {
  for (let i = 0; i < count; i += 1) {
    spawnParticle(x, y, {
      speed: random(50, 320),
      color,
      size: random(1.6, 4.8),
      life: random(0.18, 0.8)
    });
  }
}

function spawnParticle(x, y, options) {
  const angle = random(0, Math.PI * 2);
  state.entities.particles.push({
    x,
    y,
    vx: Math.cos(angle) * options.speed,
    vy: Math.sin(angle) * options.speed,
    size: options.size,
    life: options.life,
    maxLife: options.life,
    color: options.color
  });
}

function roundedRect(x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + width, y, x + width, y + height, radius);
  ctx.arcTo(x + width, y + height, x, y + height, radius);
  ctx.arcTo(x, y + height, x, y, radius);
  ctx.arcTo(x, y, x + width, y, radius);
  ctx.closePath();
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function random(min, max) {
  return Math.random() * (max - min) + min;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function inBounds(x, y, margin = 0) {
  return x >= -margin && y >= -margin && x <= canvas.width + margin && y <= canvas.height + margin;
}

function approach(value, target, amount) {
  if (value < target) {
    return Math.min(target, value + amount);
  }
  return Math.max(target, value - amount);
}

function mapRange(value, inMin, inMax, outMin, outMax) {
  if (inMax === inMin) {
    return outMin;
  }
  const t = (value - inMin) / (inMax - inMin);
  return outMin + (outMax - outMin) * t;
}
