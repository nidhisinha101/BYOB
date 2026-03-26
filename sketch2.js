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
const MOVEMENT_THRESHOLD = 75;
const MOVEMENT_MEDIUM = 150;
const MOVEMENT_FAST = 300;
const RIPPLE_SMALL_MULT = 1.5;
const RIPPLE_MEDIUM_MULT = 3.0;

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

const BODY_PART_COLORS = {
  leftShoulder: 0,
  rightShoulder: 1,
  leftElbow: 2,
  rightElbow: 3,
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
    { name: "leftElbow", hue: 10, sat: 100, count: 2 },
    { name: "rightElbow", hue: 340, sat: 100, count: 2 },
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

    colorMode(HSL);
    initializeRipples();
    initializeColors();
    
    song = loadSound('music3.mp3');
    const playButton = createButton('Get your groove on');
    playButton.parent('controls');
    playButton.addClass('play-button');
    playButton.mousePressed(() => {
      if (song.isPlaying()) {
        song.pause();
        playButton.html('Get your groove on');
      } else {
        song.loop();
        playButton.html('Pause the party');
      }
    });

    poseNet = ml5.poseNet(videoSmall, { maxPoseDetections: 5 }, modelReady);
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
    updateRipples();
    drawRipples();
    updateMovement();
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
  let compareFrame = pastPoses[max(0, pastPoses.length - 10)];
  
  if (!currentFrame || !compareFrame || currentFrame.people.length === 0) return;
  
  let CONFIDENCE_THRESHOLD = 0.5;
  
  let movements = {
    rightElbow: 0,
    leftElbow: 0,
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

    p = pose[KEYPOINT_INDICES.rightElbow];
    pp = prevPose[KEYPOINT_INDICES.rightElbow];
    if (p && pp && p.score > CONFIDENCE_THRESHOLD) {
      movement = dist(p.position.x, p.position.y, pp.position.x, pp.position.y);
      movements.rightElbow = max(movements.rightElbow, movement);
    }

    p = pose[KEYPOINT_INDICES.leftElbow];
    pp = prevPose[KEYPOINT_INDICES.leftElbow];
    if (p && pp && p.score > CONFIDENCE_THRESHOLD) {
      movement = dist(p.position.x, p.position.y, pp.position.x, pp.position.y);
      movements.leftElbow = max(movements.leftElbow, movement);
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
    
    if (movement > MOVEMENT_THRESHOLD) {
      let newSize;
      
      if (movement < MOVEMENT_MEDIUM) {
        newSize = map(movement, MOVEMENT_THRESHOLD, MOVEMENT_MEDIUM, RIPPLE_MIN * RIPPLE_SMALL_MULT, RIPPLE_MIN * RIPPLE_MEDIUM_MULT);
      } else if (movement < MOVEMENT_FAST) {
        newSize = map(movement, MOVEMENT_MEDIUM, MOVEMENT_FAST, RIPPLE_MIN * RIPPLE_MEDIUM_MULT, RIPPLE_MAX);
      } else {
        newSize = RIPPLE_MAX;
      }
      
      newSize = constrain(newSize, RIPPLE_MIN, RIPPLE_MAX);
      
      if (newSize > ripple.targetRadius) {
        ripple.targetRadius = newSize;
      }
    }
  }
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


