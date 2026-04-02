let video;
let videoSmall;
let poseNet;
let poses = [];
let pastPoses = [];
let maxPoses = 15;
let song;

let bounceSpeed = 3;
let ripples = [];
let videoAspect;
let videoOffsetX = 0;
let videoOffsetY = 0;
let videoScaleX = 1;
let videoScaleY = 1;

const MIN_SIZE = 2;
const MAX_SIZE = 150;
const RIPPLE_MIN = 20;
const RIPPLE_MAX = 350;
const MOVEMENT_THRESHOLD = 8;   // ignore micro-movements, fidgeting, nose touches
const MOVEMENT_GROOVING = 35;   // real dancing threshold — full-arm swings, body shifts
const RIPPLE_SMALL_MULT = 1.5;  // small movement → modest ripple
const RIPPLE_MEDIUM_MULT = 4.0;

const REWARD_SUSTAIN_THRESHOLD = 5;
const SPARKLE_THRESHOLD = 7;
const CONFETTI_SPAWN_INTERVAL = 300;
const SPARKLE_SPAWN_INTERVAL = 300;

const TARGET_FPS = 30;

const KEYPOINT_INDICES = {
  nose: 0,
  leftShoulder: 5,
  rightShoulder: 6,
  leftElbow: 7,
  rightElbow: 8,
  leftWrist: 9,
  rightWrist: 10,
  leftHip: 11,
  rightHip: 12,
  leftKnee: 13,
  rightKnee: 14,
};

let rippleColors = [];
let vignetteAlpha;

let rewardState = {
  currentIntensity: 'idle',
  sustainedTime: 0,
  lastIntensity: 'idle',
  rewardTriggered: false,
  rewardType: null,
  rewardTimer: 0,
};

let confetti = [];
let sparkleBursts = [];

const CONFETTI_COLORS = [
  { h: 45, s: 100, l: 60 },
  { h: 280, s: 100, l: 60 },
  { h: 180, s: 100, l: 50 },
  { h: 340, s: 100, l: 60 },
  { h: 10, s: 100, l: 55 },
  { h: 200, s: 100, l: 55 },
];

let lastConfettiSpawn = 0;
let lastSparkleSpawn = 0;
let accumulatedGrooveTime = 0;

const BODY_PART_COLORS = {
  leftShoulder: 0,
  rightShoulder: 1,
  leftWrist: 2,
  rightWrist: 3,
  leftHip: 4,
  rightHip: 5,
  leftKnee: 6,
  rightKnee: 7,
};

function initializeRipples() {
  ripples = [];
  const bodyParts = [
    { name: "leftShoulder",  hue: 180, sat: 100 },
    { name: "rightShoulder", hue: 280, sat: 100 },
    { name: "leftWrist",     hue: 10,  sat: 100 },
    { name: "rightWrist",    hue: 340, sat: 100 },
    { name: "rightHip",      hue: 160, sat: 100 },
    { name: "leftHip",       hue: 100, sat: 100 },
    { name: "rightKnee",     hue: 200, sat: 100 },
    { name: "leftKnee",      hue: 240, sat: 100 },
  ];

  for (let part of bodyParts) {
    ripples.push({
      x: width / 2,
      y: height / 2,
      targetX: width / 2,   // where the joint actually is on screen
      targetY: height / 2,
      currentRadius: RIPPLE_MIN,
      targetRadius: RIPPLE_MIN,
      hue: part.hue,
      saturation: part.sat,
      bodyPart: part.name,
      confidence: 0,        // last known confidence for this joint
    });
  }
}

function initializeColors() {
  rippleColors = [
    { stroke1: color(190, 100, 55, 200), stroke2: color(190, 100, 70, 130), stroke3: color(190, 100, 85, 80) },
    { stroke1: color(320, 100, 55, 200), stroke2: color(320, 100, 70, 130), stroke3: color(320, 100, 85, 80) },
    { stroke1: color(340, 100, 55, 200), stroke2: color(340, 100, 70, 130), stroke3: color(340, 100, 85, 80) },
    { stroke1: color(10, 100, 55, 200), stroke2: color(10, 100, 70, 130), stroke3: color(10, 100, 85, 80) },
    { stroke1: color(45, 100, 55, 200), stroke2: color(45, 100, 70, 130), stroke3: color(45, 100, 85, 80) },
    { stroke1: color(280, 100, 55, 200), stroke2: color(280, 100, 70, 130), stroke3: color(280, 100, 85, 80) },
    { stroke1: color(180, 100, 55, 200), stroke2: color(180, 100, 70, 130), stroke3: color(180, 100, 85, 80) },
    { stroke1: color(160, 100, 55, 200), stroke2: color(160, 100, 70, 130), stroke3: color(160, 100, 85, 80) },
    { stroke1: color(100, 100, 55, 200), stroke2: color(100, 100, 70, 130), stroke3: color(100, 100, 85, 80) },
  ];
  vignetteAlpha = color(10, 5, 20);
}

function setup() {
    createCanvas(windowWidth, windowHeight);
    frameRate(TARGET_FPS);
    noCursor();
    
    video = createCapture(VIDEO);
    video.size(640, 480);
    video.style('transform', 'scale(-1, 1)');
    video.hide();
    
    videoSmall = createCapture(VIDEO);
    videoSmall.size(320, 240);
    videoSmall.hide();
    
    videoAspect = 640 / 480;

    // Offscreen canvas for optical flow pixel diffing
    flowCanvas = document.createElement('canvas');
    flowCanvas.width = 80;
    flowCanvas.height = 60;
    flowCtx = flowCanvas.getContext('2d', { willReadFrequently: true });

    colorMode(HSL);
    initializeRipples();
    initializeColors();

    poseNet = ml5.poseNet(videoSmall, {
      maxPoseDetections: 5,
      minPoseConfidence: 0.1,
      minPartConfidence: 0.1,
      scoreThreshold: 0.1,
    }, modelReady);
    poseNet.on('pose', function (results) {
        poses = results;
        
        let frameData = {
          timestamp: millis(),
          people: []
        };
        
        for (let i = 0; i < poses.length; i++) {
          let keypoints = [];
          for (let j = 0; j < poses[i].pose.keypoints.length; j++) {
            keypoints.push({
              position: { x: poses[i].pose.keypoints[j].position.x * 2, y: poses[i].pose.keypoints[j].position.y * 2 },
              score: poses[i].pose.keypoints[j].score
            });
          }
          frameData.people.push({
            id: i,
            keypoints: keypoints
          });
        }
        
        pastPoses.push(frameData);
        
        while (pastPoses.length > maxPoses) {
          pastPoses.shift();
        }
    });
}

let lastCleanup = 0;
const CLEANUP_INTERVAL = 60000;

// Optical flow fallback for low-light tracking
let flowCanvas, flowCtx;
let prevPixels = null;
let flowEnergy = 0;
const FLOW_DECAY = 0.80;
const FLOW_THRESHOLD = 35;   // raised to ignore camera noise/grain in dark environments
const FLOW_SCALE = 0.0015;   // reduced so flow only nudges, not grows ripples on its own

function computeOpticalFlow() {
  if (!flowCtx || !video.elt) return;

  // Draw the current video frame down to 80x60
  flowCtx.drawImage(video.elt, 0, 0, 80, 60);
  let imageData = flowCtx.getImageData(0, 0, 80, 60);
  let pixels = imageData.data;

  if (prevPixels === null) {
    prevPixels = new Uint8ClampedArray(pixels);
    return;
  }

  let diffSum = 0;
  let count = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    let dr = abs(pixels[i]     - prevPixels[i]);
    let dg = abs(pixels[i + 1] - prevPixels[i + 1]);
    let db = abs(pixels[i + 2] - prevPixels[i + 2]);
    let diff = (dr + dg + db) / 3;
    if (diff > FLOW_THRESHOLD) {
      diffSum += diff;
      count++;
    }
  }

  // flowEnergy is 0–1 range: ratio of changed pixels × avg change magnitude
  let rawEnergy = count > 0 ? (diffSum / count) * (count / (80 * 60)) : 0;
  flowEnergy = lerp(flowEnergy, rawEnergy * FLOW_SCALE, 0.4);
  flowEnergy *= FLOW_DECAY;

  prevPixels.set(pixels);
}

function windowResized() {
    resizeCanvas(windowWidth, windowHeight);
}

function modelReady() {
    console.log('PoseNet Model Loaded!');
}

function cleanupMemory() {
    if (millis() - lastCleanup > CLEANUP_INTERVAL) {
        poses = [];
        lastCleanup = millis();
    }
}

function draw() {
    background(5, 2, 15);
    
    if (video.width > 0 && video.height > 0) {
        let canvasAspect = width / height;
        let drawW, drawH, drawX, drawY;
        
        if (videoAspect > canvasAspect) {
            drawW = width;
            drawH = width / videoAspect;
            drawX = 0;
            drawY = (height - drawH) / 2;
        } else {
            drawH = height;
            drawW = height * videoAspect;
            drawX = (width - drawW) / 2;
            drawY = 0;
        }
        
        videoOffsetX = drawX;
        videoOffsetY = drawY;
        videoScaleX = drawW / video.width;
        videoScaleY = drawH / video.height;
        
        image(video, drawX, drawY, drawW, drawH);
        
        blendMode(MULTIPLY);
        noStroke();
        fill(270, 70, 30);
        rect(drawX, drawY, drawW, drawH);
        blendMode(BLEND);
    }
    
    drawVignette();
    computeOpticalFlow();
    updateRipples();
    drawRipples();
    updateMovement();
    updateConfetti();
    updateSparkleBursts();
    drawConfetti();
    drawSparkleBursts();
    cleanupMemory();
}

function drawVignette() {
  let pulse = 0.9 + sin(frameCount * 0.02) * 0.1;
  
  noFill();
  stroke(vignetteAlpha);
  for (let i = 0; i < 10; i++) {
    strokeWeight(100);
    let alpha = map(i, 0, 10, 0.4, 0) * pulse;
    stroke(10, 5, 20, alpha * 255);
    rect(0, 0, width, height, 200);
  }
}

function updateRipples() {
  // Compute centroid of visible joints for lost-joint fallback
  let centroidX = width / 2, centroidY = height * 0.45;
  let confidentCount = 0;
  for (let r of ripples) {
    if (r.confidence > 0.25) {
      centroidX += r.targetX;
      centroidY += r.targetY;
      confidentCount++;
    }
  }
  if (confidentCount > 0) {
    centroidX /= confidentCount + 1;
    centroidY /= confidentCount + 1;
  }

  for (let ripple of ripples) {
    // Hard visibility cutoff — below this, the ripple is effectively hidden
    ripple.visible = ripple.confidence > 0.15;

    if (ripple.visible) {
      let posLerp = map(ripple.confidence, 0.15, 1, 0.06, 0.22);
      ripple.x = lerp(ripple.x, ripple.targetX, posLerp);
      ripple.y = lerp(ripple.y, ripple.targetY, posLerp);
    } else {
      // Pull quickly to centroid so if it becomes visible again it's near the body
      ripple.x = lerp(ripple.x, centroidX, 0.12);
      ripple.y = lerp(ripple.y, centroidY, 0.12);
      // Collapse radius to zero so it won't pop large when joint reappears
      ripple.targetRadius = RIPPLE_MIN;
      ripple.currentRadius = lerp(ripple.currentRadius, 0, 0.15);
    }

    if (ripple.visible) {
      let pulse = sin(frameCount * 0.04 + ripple.hue * 0.05) * 3;
      ripple.currentRadius = lerp(ripple.currentRadius, ripple.targetRadius + pulse, 0.15);
      ripple.targetRadius *= 0.96;
      ripple.targetRadius = max(ripple.targetRadius, RIPPLE_MIN);
    }
  }
}

function updateMovement() {
  if (pastPoses.length < 5) return;

  let currentFrame = pastPoses[pastPoses.length - 1];
  let compareFrame = pastPoses[max(0, pastPoses.length - 3)];

  if (!currentFrame || !compareFrame || currentFrame.people.length === 0) return;

  let CONFIDENCE_THRESHOLD = 0.25;

  // movements: pixel distance traveled since compareFrame
  let movements = {
    rightWrist: 0, leftWrist: 0,
    rightShoulder: 0, leftShoulder: 0,
    rightHip: 0, leftHip: 0,
    rightKnee: 0, leftKnee: 0,
  };

  // jointPositions: best screen-space position for each joint this frame
  let jointPositions = {};
  let jointConfidence = {};

  for (let person of currentFrame.people) {
    let personId = person.id;
    let comparePerson = compareFrame.people.find(p => p.id === personId);
    if (!comparePerson) continue;

    let pose = person.keypoints;
    let prevPose = comparePerson.keypoints;

    function processJoint(name, idx, shuffleAxis) {
      let p  = pose[idx];
      let pp = prevPose[idx];
      if (!p || !pp || p.score < CONFIDENCE_THRESHOLD) return;

      // Keypoints come from the 320x240 videoSmall, scaled *2 in the pose handler = 640x480 space.
      // The video is CSS-mirrored, so we flip x relative to the 640 width.
      let rawX = 640 - p.position.x;
      let rawY = p.position.y;

      // Map from video space to canvas space using the offsets computed in draw()
      let screenX = videoOffsetX + rawX * videoScaleX;
      let screenY = videoOffsetY + rawY * videoScaleY;

      let movement = dist(p.position.x, p.position.y, pp.position.x, pp.position.y);
      if (shuffleAxis) movement += abs(p.position.x - pp.position.x) * 1.5;

      if (movement > movements[name]) movements[name] = movement;

      // Keep highest-confidence position reading per joint
      if (!jointConfidence[name] || p.score > jointConfidence[name]) {
        jointPositions[name]  = { x: screenX, y: screenY };
        jointConfidence[name] = p.score;
      }
    }

    processJoint('rightWrist',    KEYPOINT_INDICES.rightWrist,    false);
    processJoint('leftWrist',     KEYPOINT_INDICES.leftWrist,     false);
    processJoint('rightShoulder', KEYPOINT_INDICES.rightShoulder, false);
    processJoint('leftShoulder',  KEYPOINT_INDICES.leftShoulder,  false);
    processJoint('rightHip',      KEYPOINT_INDICES.rightHip,      true);
    processJoint('leftHip',       KEYPOINT_INDICES.leftHip,       true);
    processJoint('rightKnee',     KEYPOINT_INDICES.rightKnee,     true);
    processJoint('leftKnee',      KEYPOINT_INDICES.leftKnee,      true);
  }

  // Push positions and movement into ripples
  for (let ripple of ripples) {
    let part = ripple.bodyPart;

    // Update position target if we got a reading
    if (jointPositions[part]) {
      ripple.targetX    = jointPositions[part].x;
      ripple.targetY    = jointPositions[part].y;
      ripple.confidence = jointConfidence[part];
    } else {
      ripple.confidence *= 0.75; // fast decay: joint gone for ~10 frames = invisible
    }

    let movement = movements[part];

    if (movement > MOVEMENT_THRESHOLD) {
      // Exponential curve: small movements stay small, big dancing moves explode
      // Normalize 0–1 across the full range then square it for non-linearity
      let t = constrain((movement - MOVEMENT_THRESHOLD) / (MOVEMENT_GROOVING * 3 - MOVEMENT_THRESHOLD), 0, 1);
      let tCurved = t * t * t; // cubic — tiny moves barely register, full-body = huge
      let newSize = map(tCurved, 0, 1, RIPPLE_MIN * RIPPLE_SMALL_MULT, RIPPLE_MAX);

      if (newSize > ripple.targetRadius) ripple.targetRadius = newSize;
    }
    // No flow-based growth when still — prevents noise triggering ripples unprompted
  }

  updateRewardSystem(movements);
}


function drawRipples() {
  blendMode(ADD);
  
  for (let ripple of ripples) {
    if (!ripple.visible || ripple.currentRadius < 2) continue;

    let colorIndex = BODY_PART_COLORS[ripple.bodyPart];
    let colors = rippleColors[colorIndex];
    if (!colors) continue;

    // Fade alpha based on confidence so low-confidence joints ghost out smoothly
    let alpha = map(ripple.confidence, 0.15, 0.6, 60, 255);
    alpha = constrain(alpha, 0, 255);

    let r = ripple.currentRadius;

    stroke(hue(colors.stroke1), saturation(colors.stroke1), lightness(colors.stroke1), alpha);
    strokeWeight(8);
    noFill();
    ellipse(ripple.x, ripple.y, r * 2, r * 2);

    stroke(hue(colors.stroke2), saturation(colors.stroke2), lightness(colors.stroke2), alpha * 0.65);
    strokeWeight(5);
    ellipse(ripple.x, ripple.y, r * 1.7, r * 1.7);

    stroke(hue(colors.stroke3), saturation(colors.stroke3), lightness(colors.stroke3), alpha * 0.4);
    strokeWeight(3);
    ellipse(ripple.x, ripple.y, r * 1.4, r * 1.4);
  }
  
  blendMode(BLEND);
}
 
function getCurrentMovementIntensity(movements) {
  let maxMovement = 0;
  
  for (let part in movements) {
    if (movements[part] > maxMovement) {
      maxMovement = movements[part];
    }
  }
  
  if (maxMovement >= MOVEMENT_GROOVING) {
    return 'grooving';
  } else if (maxMovement >= MOVEMENT_THRESHOLD) {
    return 'bare_minimum';
  }
  return 'idle';
}

function updateRewardSystem(movements) {
  let currentIntensity = getCurrentMovementIntensity(movements);
  
  if (currentIntensity === rewardState.lastIntensity) {
    rewardState.sustainedTime += 1 / TARGET_FPS;
  } else {
    rewardState.sustainedTime = 0;
  }
  
  rewardState.lastIntensity = currentIntensity;
  rewardState.currentIntensity = currentIntensity;
  
  if (currentIntensity === 'grooving') {
    accumulatedGrooveTime += 1 / TARGET_FPS;
    rewardState.rewardTriggered = true;
    rewardState.rewardType = 'grooving';
    spawnRewardEffects();
  } else {
    accumulatedGrooveTime = 0;
    rewardState.rewardTriggered = false;
    rewardState.rewardType = null;
    confetti = [];
    sparkleBursts = [];
  }
}

function spawnRewardEffects() {
  let now = millis();
  
  if (accumulatedGrooveTime >= REWARD_SUSTAIN_THRESHOLD) {
    if (now - lastConfettiSpawn > CONFETTI_SPAWN_INTERVAL) {
      spawnConfetti(4);
      lastConfettiSpawn = now;
    }
  }
  
  if (accumulatedGrooveTime >= SPARKLE_THRESHOLD) {
    if (now - lastSparkleSpawn > SPARKLE_SPAWN_INTERVAL) {
      spawnSparkleBurst();
      lastSparkleSpawn = now;
    }
  }
}

function spawnConfetti(count) {
  for (let i = 0; i < count; i++) {
    let col = random(CONFETTI_COLORS);
    confetti.push({
      x: random(width),
      y: -20,
      size: random(8, 16),
      rotation: random(TWO_PI),
      rotationSpeed: random(-0.2, 0.2),
      vx: random(-1, 1),
      vy: random(2, 5),
      h: col.h,
      s: col.s,
      l: col.l,
      lifetime: random(3, 5),
      age: 0,
    });
  }
}

function spawnSparkleBurst() {
  let centerX = random(width * 0.2, width * 0.8);
  let centerY = random(height * 0.2, height * 0.7);
  let particleCount = floor(random(8, 15));
  let col = random(CONFETTI_COLORS);
  
  for (let i = 0; i < particleCount; i++) {
    let angle = random(TWO_PI);
    let speed = random(2, 6);
    sparkleBursts.push({
      x: centerX,
      y: centerY,
      vx: cos(angle) * speed,
      vy: sin(angle) * speed,
      size: random(4, 10),
      h: col.h,
      s: col.s,
      l: col.l,
      lifetime: random(0.8, 1.5),
      age: 0,
    });
  }
}

function updateConfetti() {
  for (let i = confetti.length - 1; i >= 0; i--) {
    let c = confetti[i];
    c.x += c.vx + sin(frameCount * 0.05 + i) * 0.5;
    c.y += c.vy;
    c.rotation += c.rotationSpeed;
    c.age += 1 / TARGET_FPS;
    
    if (c.age > c.lifetime || c.y > height + 50) {
      confetti.splice(i, 1);
    }
  }
}

function updateSparkleBursts() {
  for (let i = sparkleBursts.length - 1; i >= 0; i--) {
    let s = sparkleBursts[i];
    s.x += s.vx;
    s.y += s.vy;
    s.vx *= 0.95;
    s.vy *= 0.95;
    s.age += 1 / TARGET_FPS;
    
    if (s.age > s.lifetime) {
      sparkleBursts.splice(i, 1);
    }
  }
}

function drawConfetti() {
  for (let c of confetti) {
    let fadeAlpha = 1;
    if (c.age > c.lifetime - 1) {
      fadeAlpha = (c.lifetime - c.age);
    }
    
    push();
    translate(c.x, c.y);
    rotate(c.rotation);
    noStroke();
    fill(c.h, c.s, c.l, fadeAlpha * 255);
    rectMode(CENTER);
    rect(0, 0, c.size, c.size * 0.6);
    pop();
  }
}

function drawSparkleBursts() {
  for (let s of sparkleBursts) {
    let fadeAlpha = 1;
    if (s.age > s.lifetime - 0.3) {
      fadeAlpha = (s.lifetime - s.age) / 0.3;
    }
    
    push();
    noStroke();
    fill(s.h, s.s, s.l, fadeAlpha * 255);
    translate(s.x, s.y);
    
    for (let i = 0; i < 4; i++) {
      let angle = TWO_PI / 4 * i;
      push();
      rotate(angle);
      ellipse(0, -s.size / 2, s.size * 0.2, s.size);
      pop();
    }
    pop();
  }
}
