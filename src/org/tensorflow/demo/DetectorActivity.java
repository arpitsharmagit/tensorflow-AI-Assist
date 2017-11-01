/*
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.demo;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.util.Size;
import android.util.TypedValue;
import android.view.Display;
import android.widget.TextView;
import android.widget.Toast;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;
import org.tensorflow.demo.OverlayView.DrawCallback;
import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.tracking.MultiBoxTracker;
import android.os.Vibrator;
import org.tensorflow.demo.R; // Explicit import needed for internal Google builds.

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged multibox model.
  private static final int MB_INPUT_SIZE = 224;
  private static final int MB_IMAGE_MEAN = 128;
  private static final float MB_IMAGE_STD = 128;
  private static final String MB_INPUT_NAME = "ResizeBilinear";
  private static final String MB_OUTPUT_LOCATIONS_NAME = "output_locations/Reshape";
  private static final String MB_OUTPUT_SCORES_NAME = "output_scores/Reshape";
  private static final String MB_MODEL_FILE = "file:///android_asset/multibox_model.pb";
  private static final String MB_LOCATION_FILE =
      "file:///android_asset/multibox_location_priors.txt";

  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final String TF_OD_API_MODEL_FILE =
      "file:///android_asset/ssd_mobilenet_v1_android_export.pb";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/coco_labels_list.txt";

  // Configuration values for tiny-yolo-voc. Note that the graph is not included with TensorFlow and
  // must be manually placed in the assets/ directory by the user.
  // Graphs and models downloaded from http://pjreddie.com/darknet/yolo/ may be converted e.g. via
  // DarkFlow (https://github.com/thtrieu/darkflow). Sample command:
  // ./flow --model cfg/tiny-yolo-voc.cfg --load bin/tiny-yolo-voc.weights --savepb --verbalise
  private static final String YOLO_MODEL_FILE = "file:///android_asset/graph-tiny-yolo-voc.pb";
  private static final int YOLO_INPUT_SIZE = 416;
  private static final String YOLO_INPUT_NAME = "input";
  private static final String YOLO_OUTPUT_NAMES = "output";
  private static final int YOLO_BLOCK_SIZE = 32;

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.  Optionally use legacy Multibox (trained using an older version of the API)
  // or YOLO.

    private static final int ImageClassifier_INPUT_SIZE = 224;
    private static final int ImageClassifier_IMAGE_MEAN = 117;
    private static final float ImageClassifier_IMAGE_STD = 1;
    private static final String ImageClassifier_INPUT_NAME = "input";
    private static final String ImageClassifier_OUTPUT_NAME = "output";
    private static final String ImageClassifier_MODEL_FILE = "file:///android_asset/tensorflow_inception_graph.pb";
    private static final String ImageClassifier_LABEL_FILE =
            "file:///android_asset/imagenet_comp_graph_label_strings.txt";
  private enum DetectorMode {
    TF_OD_API, MULTIBOX, YOLO,ImageClassifier;
  }
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;

  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;
  private static final float MINIMUM_CONFIDENCE_MULTIBOX = 0.1f;
  private static final float MINIMUM_CONFIDENCE_YOLO = 0.25f;
    private static final float MINIMUM_CONFIDENCE_IMAGECLASSIFIER = 0.1f;
    private static final float FOV = 125f;
  private static final boolean MAINTAIN_ASPECT = MODE == DetectorMode.YOLO;

  private static final Size DESIRED_PREVIEW_SIZE = new Size(640,480 );

  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;

  private Integer sensorOrientation;

  private Classifier detector;
  private Speaker speaker;
  private TextView textViewResult=null;
  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;
    private long lastDetectTimestamp =0;
    private  String currentObjectTitle = "";
    private List<String> identifiableObjects;
    private Vibrator vibrator;
  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private byte[] luminanceCopy;

  private BorderedText borderedText;
  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
      LOGGER.i("Display: "+getResources().getDisplayMetrics().toString());
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;
    if (MODE == DetectorMode.YOLO) {
      detector =
          TensorFlowYoloDetector.create(
              getAssets(),
              YOLO_MODEL_FILE,
              YOLO_INPUT_SIZE,
              YOLO_INPUT_NAME,
              YOLO_OUTPUT_NAMES,
              YOLO_BLOCK_SIZE);
      cropSize = YOLO_INPUT_SIZE;
    } else if (MODE == DetectorMode.MULTIBOX) {
      detector =
          TensorFlowMultiBoxDetector.create(
              getAssets(),
              MB_MODEL_FILE,
              MB_LOCATION_FILE,
              MB_IMAGE_MEAN,
              MB_IMAGE_STD,
              MB_INPUT_NAME,
              MB_OUTPUT_LOCATIONS_NAME,
              MB_OUTPUT_SCORES_NAME);
      cropSize = MB_INPUT_SIZE;
    } if(MODE==DetectorMode.ImageClassifier){
          detector =
                  TensorFlowImageClassifier.create(
                          getAssets(),
                          ImageClassifier_MODEL_FILE,
                          ImageClassifier_LABEL_FILE,
                          ImageClassifier_INPUT_SIZE,
                          ImageClassifier_IMAGE_MEAN,
                          ImageClassifier_IMAGE_STD,
                          ImageClassifier_INPUT_NAME,
                          ImageClassifier_OUTPUT_NAME);
          cropSize =ImageClassifier_INPUT_SIZE;

      } else {
      try {
        detector = TensorFlowObjectDetectionAPIModel.create(
            getAssets(), TF_OD_API_MODEL_FILE, TF_OD_API_LABELS_FILE, TF_OD_API_INPUT_SIZE);
        cropSize = TF_OD_API_INPUT_SIZE;
      } catch (final IOException e) {
        LOGGER.e("Exception initializing classifier!", e);
        Toast toast =
            Toast.makeText(
                getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
        toast.show();
        finish();
      }
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    final Display display = getWindowManager().getDefaultDisplay();
    final int screenOrientation = display.getRotation();

    LOGGER.i("Sensor orientation: %d, Screen orientation: %d", rotation, screenOrientation);

    sensorOrientation = rotation + screenOrientation;

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            tracker.draw(canvas);
            if (isDebug()) {
              tracker.drawDebug(canvas);
            }
          }
        });

    addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            if (!isDebug()) {
              return;
            }
            final Bitmap copy = cropCopyBitmap;
            if (copy == null) {
              return;
            }

            final int backgroundColor = Color.argb(100, 0, 0, 0);
            canvas.drawColor(backgroundColor);

            final Matrix matrix = new Matrix();
            final float scaleFactor = 2;
            matrix.postScale(scaleFactor, scaleFactor);
            matrix.postTranslate(
                canvas.getWidth() - copy.getWidth() * scaleFactor,
                canvas.getHeight() - copy.getHeight() * scaleFactor);
            canvas.drawBitmap(copy, matrix, new Paint());

            final Vector<String> lines = new Vector<String>();
            if (detector != null) {
              final String statString = detector.getStatString();
              final String[] statLines = statString.split("\n");
              for (final String line : statLines) {
                lines.add(line);
              }
            }
            lines.add("");

            lines.add("Frame: " + previewWidth + "x" + previewHeight);
            lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
            lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
            lines.add("Rotation: " + sensorOrientation);
            lines.add("Inference time: " + lastProcessingTimeMs + "ms");

            borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
          }
        });
  }

  OverlayView trackingOverlay;
  @Override
  protected void onCreate(final Bundle savedInstanceState) {
    LOGGER.d("onCreate " + this);
    super.onCreate(null);
    vibrator = (Vibrator) this.getSystemService(Context.VIBRATOR_SERVICE);
    identifiableObjects = Arrays.asList("chair","stop sign","stone wall");
    speaker = new Speaker(this);
    speaker.allow(true);
  }
  @Override
  public synchronized void onDestroy() {
    LOGGER.d("onDestroy " + this);
    super.onDestroy();
    if (speaker != null) {
      speaker.destroy();
    }
  }
  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;

    byte[] originalLuminance = getLuminance();
    tracker.onFrame(
        previewWidth,
        previewHeight,
        getLuminanceStride(),
        sensorOrientation,
        originalLuminance,
        timestamp);
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    // LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    if (luminanceCopy == null) {
      luminanceCopy = new byte[originalLuminance.length];
    }
    System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);
    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            // LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();
            final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);

            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
            switch (MODE) {
              case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
              case MULTIBOX:
                minimumConfidence = MINIMUM_CONFIDENCE_MULTIBOX;
                break;
              case YOLO:
                minimumConfidence = MINIMUM_CONFIDENCE_YOLO;
                break;
                case ImageClassifier:
                    minimumConfidence =MINIMUM_CONFIDENCE_IMAGECLASSIFIER;
                    break;
            }

            final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();
            boolean haschair = false;boolean IsInRange=false;boolean isObjectOnLeft =false;boolean isObjectOnRight =false;
            Classifier.Recognition _result  = null;

            for (final Classifier.Recognition result : results) {
                final RectF location = result.getLocation();
                LOGGER.i(result.getTitle() + " " + result.getConfidence()+ " "+ location);
                if (location != null && result.getConfidence() >= minimumConfidence) {
                    canvas.drawRect(location, paint);
                    if (isValidArea(location) && identifiableObjects.contains(result.getTitle())
                            && result.getConfidence() >= minimumConfidence) {
                        IsInRange = isInRange(location);
                        isObjectOnLeft = isInLeft(location);
                        LOGGER.i(result.getTitle()+" OnLeft: "+isObjectOnLeft);
                        isObjectOnRight = isInRight(location);
                        LOGGER.i(result.getTitle()+" OnRight: "+isObjectOnRight);
                        _result = result;
                    }
                    cropToFrameTransform.mapRect(location);
                    result.setLocation(location);
                    mappedRecognitions.add(result);
                }
            }
             if(_result != null) {
                 if (_result.getTitle().equals("chair")) {
                     haschair = true;
                 }
                 LOGGER.i("InRange: "+IsInRange);
                 LOGGER.i("isPreviousImage: "+!_result.getTitle().equals(currentObjectTitle));
                 LOGGER.i("InIdentifiableObject: "+identifiableObjects.contains(_result.getTitle()));
                 LOGGER.i("Confidence: "+(_result.getConfidence() >= 0.60));
                 LOGGER.i(" ");
                 if (IsInRange && !_result.getTitle().equals(currentObjectTitle)
                         && identifiableObjects.contains(_result.getTitle())
                         && _result.getConfidence() >= 0.60) {
                     currentObjectTitle = _result.getTitle();
                     vibrator.vibrate(500);
                     String text = currentObjectTitle;
                     if(isObjectOnLeft){
                         text= text.concat(" on your left");
                     }else if(isObjectOnRight){
                         text= text.concat(" on your right");
                     }
                     else{
                        text=  text.concat(" in your way");
                     }
                     speaker.speak(text);
                 }
             }
             long differenceMillis =SystemClock.uptimeMillis() - lastDetectTimestamp;
              if(!haschair && currentObjectTitle.equals("chair")&&differenceMillis>6000 ){
                  lastDetectTimestamp = SystemClock.uptimeMillis();
                  currentObjectTitle = "";
                  speaker.speak("chair removed");
              }

            tracker.trackResults(mappedRecognitions, luminanceCopy, currTimestamp);
            trackingOverlay.postInvalidate();

            requestRender();
            computingDetection = false;
          }
        });
  }

    private boolean isInRange(RectF location) {
      int width =getResources().getDisplayMetrics().widthPixels;
        int height=getResources().getDisplayMetrics().heightPixels;
        LOGGER.i("Display: w: %s, h: %s",width,height);
      LOGGER.i("DetectionSize: w:%s ,h: %s",Math.round(location.width()),Math.round(location.height()));
//        if( location.left > 5 &&  location.right <295){
//            return true;
//        }
//        return false;
        return  true;
    }
    private boolean isInLeft(RectF location){
      if(location.left>0 && location.right<150){
          return true;
      }
      return false;
    }
    private boolean isInRight(RectF location){
        if(location.left>150 && location.right<300){
            return true;
        }
        return false;
    }
    private boolean isValidArea(RectF location) {
      double objectLength = location.right - location.left;
        double objectWidth = location.bottom - location.top;
        double objectArea = Math.round(objectLength* objectWidth);

        if(objectArea >  5000){
            LOGGER.i("Left: %s, Right: %s",Math.round(location.left),Math.round(location.right));
            LOGGER.i("Area: w: %s, h: %s",location.width(),location.height());
            LOGGER.i("Area Is " + objectArea);
        }
        return objectArea >  5000;
    }
    private int distance(RectF location) {
        double faceWidthInPX =  location.right - location.left;
        double faceWidthInCm = 17;
        float viewWiddthINPX = 300;
        double viewWidthInCM = (faceWidthInCm * viewWiddthINPX)/ faceWidthInPX;
        double Focaldegree = Math.tan(FOV/2);
        double  dis = ((faceWidthInCm * viewWiddthINPX))/(faceWidthInPX*2*Focaldegree);
        return  (int)Math.round(dis<0?dis*-1: dis);
    }

    @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  @Override
  public void onSetDebug(final boolean debug) {
    detector.enableStatLogging(debug);
  }
}
