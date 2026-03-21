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
const MOVEMENT_THRESHOLD = 15;

const TARGET_FPS = 30;

const KEYPOINT_INDICES = {
  nose: 0,
  leftEye: 1,
  rightEye: 2,
  leftEar: 3,
  rightEar: 4,
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
  leftAnkle: 15,
  rightAnkle: 16
};

let rippleColors = [];
let vignetteAlpha;

const BODY_PART_COLORS = {
  rightHand: 0,
  leftHand: 1,
  rightElbow: 2,
  leftElbow: 3,
  nose: 4,
  rightShoulder: 5,
  leftShoulder: 6,
  rightHip: 7,
  leftHip: 8,
  rightKnee: 9,
  leftKnee: 10,
  rightAnkle: 11,
  leftAnkle: 12
};

function initializeRipples() {
  ripples = [];
  const bodyParts = [
    { name: "rightHand", hue: 190, sat: 100, count: 2 },
    { name: "leftHand", hue: 320, sat: 100, count: 2 },
    { name: "rightElbow", hue: 340, sat: 100, count: 2 },
    { name: "leftElbow", hue: 10, sat: 100, count: 2 },
    { name: "nose", hue: 45, sat: 100, count: 2 },
    { name: "rightShoulder", hue: 280, sat: 100, count: 2 },
    { name: "leftShoulder", hue: 180, sat: 100, count: 2 },
    { name: "rightHip", hue: 160, sat: 100, count: 2 },
    { name: "leftHip", hue: 100, sat: 100, count: 2 },
    { name: "rightKnee", hue: 200, sat: 100, count: 2 },
    { name: "leftKnee", hue: 240, sat: 100, count: 2 },
    { name: "rightAnkle", hue: 220, sat: 100, count: 2 },
    { name: "leftAnkle", hue: 260, sat: 100, count: 2 },
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
    { stroke1: color(200, 100, 55, 200), stroke2: color(200, 100, 70, 130), stroke3: color(200, 100, 85, 80) },
    { stroke1: color(240, 100, 55, 200), stroke2: color(240, 100, 70, 130), stroke3: color(240, 100, 85, 80) },
    { stroke1: color(220, 100, 55, 200), stroke2: color(220, 100, 70, 130), stroke3: color(220, 100, 85, 80) },
    { stroke1: color(260, 100, 55, 200), stroke2: color(260, 100, 70, 130), stroke3: color(260, 100, 85, 80) },
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
    rightHand: 0,
    leftHand: 0,
    rightElbow: 0,
    leftElbow: 0,
    nose: 0,
    rightShoulder: 0,
    leftShoulder: 0,
    rightHip: 0,
    leftHip: 0,
    rightKnee: 0,
    leftKnee: 0,
    rightAnkle: 0,
    leftAnkle: 0
  };
  
  for (let person of currentFrame.people) {
    let personId = person.id;
    let comparePerson = compareFrame.people.find(p => p.id === personId);
    
    if (!comparePerson) continue;
    
    let pose = person.keypoints;
    let prevPose = comparePerson.keypoints;
    
    if (pose[KEYPOINT_INDICES.rightWrist].score > CONFIDENCE_THRESHOLD) {
      let move = dist(pose[KEYPOINT_INDICES.rightWrist].position.x, pose[KEYPOINT_INDICES.rightWrist].position.y, 
                      prevPose[KEYPOINT_INDICES.rightWrist].position.x, prevPose[KEYPOINT_INDICES.rightWrist].position.y);
      movements.rightHand = max(movements.rightHand, move);
    }
    if (pose[KEYPOINT_INDICES.leftWrist].score > CONFIDENCE_THRESHOLD) {
      let move = dist(pose[KEYPOINT_INDICES.leftWrist].position.x, pose[KEYPOINT_INDICES.leftWrist].position.y,
                      prevPose[KEYPOINT_INDICES.leftWrist].position.x, prevPose[KEYPOINT_INDICES.leftWrist].position.y);
      movements.leftHand = max(movements.leftHand, move);
    }
    if (pose[KEYPOINT_INDICES.rightElbow].score > CONFIDENCE_THRESHOLD) {
      let move = dist(pose[KEYPOINT_INDICES.rightElbow].position.x, pose[KEYPOINT_INDICES.rightElbow].position.y,
                      prevPose[KEYPOINT_INDICES.rightElbow].position.x, prevPose[KEYPOINT_INDICES.rightElbow].position.y);
      movements.rightElbow = max(movements.rightElbow, move);
    }
    if (pose[KEYPOINT_INDICES.leftElbow].score > CONFIDENCE_THRESHOLD) {
      let move = dist(pose[KEYPOINT_INDICES.leftElbow].position.x, pose[KEYPOINT_INDICES.leftElbow].position.y,
                      prevPose[KEYPOINT_INDICES.leftElbow].position.x, prevPose[KEYPOINT_INDICES.leftElbow].position.y);
      movements.leftElbow = max(movements.leftElbow, move);
    }
    if (pose[KEYPOINT_INDICES.nose].score > CONFIDENCE_THRESHOLD) {
      let move = dist(pose[KEYPOINT_INDICES.nose].position.x, pose[KEYPOINT_INDICES.nose].position.y,
                      prevPose[KEYPOINT_INDICES.nose].position.x, prevPose[KEYPOINT_INDICES.nose].position.y);
      movements.nose = max(movements.nose, move);
    }
    if (pose[KEYPOINT_INDICES.rightShoulder].score > CONFIDENCE_THRESHOLD) {
      let move = dist(pose[KEYPOINT_INDICES.rightShoulder].position.x, pose[KEYPOINT_INDICES.rightShoulder].position.y,
                      prevPose[KEYPOINT_INDICES.rightShoulder].position.x, prevPose[KEYPOINT_INDICES.rightShoulder].position.y);
      movements.rightShoulder = max(movements.rightShoulder, move);
    }
    if (pose[KEYPOINT_INDICES.leftShoulder].score > CONFIDENCE_THRESHOLD) {
      let move = dist(pose[KEYPOINT_INDICES.leftShoulder].position.x, pose[KEYPOINT_INDICES.leftShoulder].position.y,
                      prevPose[KEYPOINT_INDICES.leftShoulder].position.x, prevPose[KEYPOINT_INDICES.leftShoulder].position.y);
      movements.leftShoulder = max(movements.leftShoulder, move);
    }
    if (pose[KEYPOINT_INDICES.rightHip].score > CONFIDENCE_THRESHOLD) {
      let hipMove = dist(pose[KEYPOINT_INDICES.rightHip].position.x, pose[KEYPOINT_INDICES.rightHip].position.y,
                        prevPose[KEYPOINT_INDICES.rightHip].position.x, prevPose[KEYPOINT_INDICES.rightHip].position.y);
      let shuffleBonus = abs(pose[KEYPOINT_INDICES.rightHip].position.x - prevPose[KEYPOINT_INDICES.rightHip].position.x) * 1.5;
      movements.rightHip = max(movements.rightHip, hipMove + shuffleBonus);
    }
    if (pose[KEYPOINT_INDICES.leftHip].score > CONFIDENCE_THRESHOLD) {
      let hipMove = dist(pose[KEYPOINT_INDICES.leftHip].position.x, pose[KEYPOINT_INDICES.leftHip].position.y,
                        prevPose[KEYPOINT_INDICES.leftHip].position.x, prevPose[KEYPOINT_INDICES.leftHip].position.y);
      let shuffleBonus = abs(pose[KEYPOINT_INDICES.leftHip].position.x - prevPose[KEYPOINT_INDICES.leftHip].position.x) * 1.5;
      movements.leftHip = max(movements.leftHip, hipMove + shuffleBonus);
    }
    if (pose[KEYPOINT_INDICES.rightKnee].score > CONFIDENCE_THRESHOLD) {
      let kneeMove = dist(pose[KEYPOINT_INDICES.rightKnee].position.x, pose[KEYPOINT_INDICES.rightKnee].position.y,
                          prevPose[KEYPOINT_INDICES.rightKnee].position.x, prevPose[KEYPOINT_INDICES.rightKnee].position.y);
      let shuffleBonus = abs(pose[KEYPOINT_INDICES.rightKnee].position.x - prevPose[KEYPOINT_INDICES.rightKnee].position.x) * 1.5;
      movements.rightKnee = max(movements.rightKnee, kneeMove + shuffleBonus);
    }
    if (pose[KEYPOINT_INDICES.leftKnee].score > CONFIDENCE_THRESHOLD) {
      let kneeMove = dist(pose[KEYPOINT_INDICES.leftKnee].position.x, pose[KEYPOINT_INDICES.leftKnee].position.y,
                          prevPose[KEYPOINT_INDICES.leftKnee].position.x, prevPose[KEYPOINT_INDICES.leftKnee].position.y);
      let shuffleBonus = abs(pose[KEYPOINT_INDICES.leftKnee].position.x - prevPose[KEYPOINT_INDICES.leftKnee].position.x) * 1.5;
      movements.leftKnee = max(movements.leftKnee, kneeMove + shuffleBonus);
    }
    if (pose[KEYPOINT_INDICES.rightAnkle].score > CONFIDENCE_THRESHOLD) {
      let ankleMove = dist(pose[KEYPOINT_INDICES.rightAnkle].position.x, pose[KEYPOINT_INDICES.rightAnkle].position.y,
                            prevPose[KEYPOINT_INDICES.rightAnkle].position.x, prevPose[KEYPOINT_INDICES.rightAnkle].position.y);
      let shuffleBonus = abs(pose[KEYPOINT_INDICES.rightAnkle].position.x - prevPose[KEYPOINT_INDICES.rightAnkle].position.x) * 1.5;
      movements.rightAnkle = max(movements.rightAnkle, ankleMove + shuffleBonus);
    }
    if (pose[KEYPOINT_INDICES.leftAnkle].score > CONFIDENCE_THRESHOLD) {
      let ankleMove = dist(pose[KEYPOINT_INDICES.leftAnkle].position.x, pose[KEYPOINT_INDICES.leftAnkle].position.y,
                            prevPose[KEYPOINT_INDICES.leftAnkle].position.x, prevPose[KEYPOINT_INDICES.leftAnkle].position.y);
      let shuffleBonus = abs(pose[KEYPOINT_INDICES.leftAnkle].position.x - prevPose[KEYPOINT_INDICES.leftAnkle].position.x) * 1.5;
      movements.leftAnkle = max(movements.leftAnkle, ankleMove + shuffleBonus);
    }
  }
  
  for (let ripple of ripples) {
    let movement = movements[ripple.bodyPart];
    
    if (movement > MOVEMENT_THRESHOLD) {
      let newSize = map(movement, MOVEMENT_THRESHOLD, 100, RIPPLE_MIN, RIPPLE_MAX);
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


