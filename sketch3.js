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
const RIPPLE_MAX = 100;
const MOVEMENT_THRESHOLD = 1;
const MOVEMENT_GROOVING = 1.5;
const RIPPLE_SMALL_MULT = 1.5;
const RIPPLE_MEDIUM_MULT = 3.0;

const REWARD_SUSTAIN_THRESHOLD = 5;
const SPARKLE_THRESHOLD = 7;
const CONFETTI_SPAWN_INTERVAL = 300;
const SPARKLE_SPAWN_INTERVAL = 300;

const TARGET_FPS = 30;

const KEYPOINT_INDICES = {
  leftShoulder: 5,
  rightShoulder: 6,
  leftElbow: 7,
  rightElbow: 8,
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
    { name: "leftShoulder", hue: 180, sat: 100, count: 2 },
    { name: "rightShoulder", hue: 280, sat: 100, count: 2 },
    { name: "leftWrist", hue: 10, sat: 100, count: 2 },
    { name: "rightWrist", hue: 340, sat: 100, count: 2 },
    { name: "rightHip", hue: 160, sat: 100, count: 2 },
    { name: "leftHip", hue: 100, sat: 100, count: 2 },
    { name: "rightKnee", hue: 200, sat: 100, count: 2 },
    { name: "leftKnee", hue: 240, sat: 100, count: 2 },
  ];
  
  for (let part of bodyParts) {
    for (let i = 0; i < part.count; i++) {
      ripples.push({
        x: random(100, width - 100),
        y: random(100, height - 100),
        vx: random(-bounceSpeed, bounceSpeed),
        vy: random(-bounceSpeed, bounceSpeed),
        currentRadius: RIPPLE_MIN,
        targetRadius: RIPPLE_MIN,
        hue: part.hue,
        saturation: part.sat,
        bodyPart: part.name
      });
    }
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
const FLOW_DECAY = 0.85;
const FLOW_THRESHOLD = 15;   // per-channel pixel diff to count as movement
const FLOW_SCALE = 0.003;    // how strongly flow energy drives ripple size

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
  const margin = 50;
  
  for (let ripple of ripples) {
    ripple.x += ripple.vx;
    ripple.y += ripple.vy;
    
    if (ripple.x < margin || ripple.x > width - margin) {
      ripple.vx *= -1;
      ripple.x = constrain(ripple.x, margin, width - margin);
    }
    if (ripple.y < margin || ripple.y > height - margin) {
      ripple.vy *= -1;
      ripple.y = constrain(ripple.y, margin, height - margin);
    }
    
    ripple.currentRadius = lerp(ripple.currentRadius, ripple.targetRadius, 0.15);
    ripple.targetRadius *= 0.97;
    ripple.targetRadius = max(ripple.targetRadius, RIPPLE_MIN);
  }
}

function updateMovement() {
  if (pastPoses.length < 5) return;
  
  let currentFrame = pastPoses[pastPoses.length - 1];
  let compareFrame = pastPoses[max(0, pastPoses.length - 3)];
  
  if (!currentFrame || !compareFrame || currentFrame.people.length === 0) return;
  
  let CONFIDENCE_THRESHOLD = 0.25;
  
  let movements = {
    rightWrist: 0,
    leftWrist: 0,
    rightShoulder: 0,
    leftShoulder: 0,
    rightHip: 0,
    leftHip: 0,
    rightKnee: 0,
    leftKnee: 0,
  };
  
  for (let person of currentFrame.people) {
    let personId = person.id;
    let comparePerson = compareFrame.people.find(p => p.id === personId);
    
    if (!comparePerson) continue;
    
    let pose = person.keypoints;
    let prevPose = comparePerson.keypoints;
    
    let p, pp, movement, shuffleBonus;

    p = pose[KEYPOINT_INDICES.rightWrist];
    pp = prevPose[KEYPOINT_INDICES.rightWrist];
    if (p && pp && p.score > CONFIDENCE_THRESHOLD) {
      movement = dist(p.position.x, p.position.y, pp.position.x, pp.position.y);
      movements.rightWrist = max(movements.rightWrist, movement);
    }

    p = pose[KEYPOINT_INDICES.leftWrist];
    pp = prevPose[KEYPOINT_INDICES.leftWrist];
    if (p && pp && p.score > CONFIDENCE_THRESHOLD) {
      movement = dist(p.position.x, p.position.y, pp.position.x, pp.position.y);
      movements.leftWrist = max(movements.leftWrist, movement);
    }

    p = pose[KEYPOINT_INDICES.rightShoulder];
    pp = prevPose[KEYPOINT_INDICES.rightShoulder];
    if (p && pp && p.score > CONFIDENCE_THRESHOLD) {
      movement = dist(p.position.x, p.position.y, pp.position.x, pp.position.y);
      movements.rightShoulder = max(movements.rightShoulder, movement);
    }

    p = pose[KEYPOINT_INDICES.leftShoulder];
    pp = prevPose[KEYPOINT_INDICES.leftShoulder];
    if (p && pp && p.score > CONFIDENCE_THRESHOLD) {
      movement = dist(p.position.x, p.position.y, pp.position.x, pp.position.y);
      movements.leftShoulder = max(movements.leftShoulder, movement);
    }

    p = pose[KEYPOINT_INDICES.rightHip];
    pp = prevPose[KEYPOINT_INDICES.rightHip];
    if (p && pp && p.score > CONFIDENCE_THRESHOLD) {
      movement = dist(p.position.x, p.position.y, pp.position.x, pp.position.y);
      shuffleBonus = abs(p.position.x - pp.position.x) * 1.5;
      movements.rightHip = max(movements.rightHip, movement + shuffleBonus);
    }

    p = pose[KEYPOINT_INDICES.leftHip];
    pp = prevPose[KEYPOINT_INDICES.leftHip];
    if (p && pp && p.score > CONFIDENCE_THRESHOLD) {
      movement = dist(p.position.x, p.position.y, pp.position.x, pp.position.y);
      shuffleBonus = abs(p.position.x - pp.position.x) * 1.5;
      movements.leftHip = max(movements.leftHip, movement + shuffleBonus);
    }

    p = pose[KEYPOINT_INDICES.rightKnee];
    pp = prevPose[KEYPOINT_INDICES.rightKnee];
    if (p && pp && p.score > CONFIDENCE_THRESHOLD) {
      movement = dist(p.position.x, p.position.y, pp.position.x, pp.position.y);
      shuffleBonus = abs(p.position.x - pp.position.x) * 1.5;
      movements.rightKnee = max(movements.rightKnee, movement + shuffleBonus);
    }

    p = pose[KEYPOINT_INDICES.leftKnee];
    pp = prevPose[KEYPOINT_INDICES.leftKnee];
    if (p && pp && p.score > CONFIDENCE_THRESHOLD) {
      movement = dist(p.position.x, p.position.y, pp.position.x, pp.position.y);
      shuffleBonus = abs(p.position.x - pp.position.x) * 1.5;
      movements.leftKnee = max(movements.leftKnee, movement + shuffleBonus);
    }
  }
  
  for (let ripple of ripples) {
    let movement = movements[ripple.bodyPart];

    // Optical flow fallback: if PoseNet is struggling (low movement readings),
    // flowEnergy keeps ripples alive based on raw pixel motion in the frame
    let flowFloor = map(flowEnergy, 0, 1, RIPPLE_MIN, RIPPLE_MIN * 4);
    
    if (movement > MOVEMENT_THRESHOLD) {
      let newSize;
      
      if (movement < MOVEMENT_GROOVING) {
        newSize = map(movement, MOVEMENT_THRESHOLD, MOVEMENT_GROOVING, RIPPLE_MIN * RIPPLE_SMALL_MULT, RIPPLE_MIN * RIPPLE_MEDIUM_MULT);
      } else {
        newSize = RIPPLE_MAX;
      }
      
      newSize = constrain(newSize, RIPPLE_MIN, RIPPLE_MAX);
      
      if (newSize > ripple.targetRadius) {
        ripple.targetRadius = newSize;
      }
    } else {
      // No confident pose data — use flow energy to keep ripples gently alive
      if (flowFloor > ripple.targetRadius) {
        ripple.targetRadius = flowFloor;
      }
    }
  }
  updateRewardSystem(movements);
}

function drawRipples() {
  blendMode(ADD);
  
  for (let ripple of ripples) {
    let colorIndex = BODY_PART_COLORS[ripple.bodyPart];
    let colors = rippleColors[colorIndex];
    if (!colors) continue;

    let r = ripple.currentRadius;
    
    stroke(colors.stroke1);
    strokeWeight(8);
    noFill();
    ellipse(ripple.x, ripple.y, r * 2, r * 2);
    
    stroke(colors.stroke2);
    strokeWeight(5);
    ellipse(ripple.x, ripple.y, r * 1.7, r * 1.7);
    
    stroke(colors.stroke3);
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
