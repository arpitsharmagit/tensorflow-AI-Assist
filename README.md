# AI Assist Application
 
This is an Android Application to help Visually/Hearing Impaired people in their daily tasks in RealTime.

## Description
It uses Google's TensorFlow pre-trained models to detect objects in video feed and provides feedback to users via Speech and Vibration APIs.


Inference is done using the TensorFlow Android Inference
Interface, which may be built separately
if you want a standalone library to drop into your existing application.

A device running Android 5.0 (API 21) or higher is required to run the demo due
to the use of the camera2 API, although the native libraries themselves can run
on API >= 14 devices.

## Prebuilt Components:

If you just want the fastest path to trying the demo, you may download the
nightly build
[here](https://ci.tensorflow.org/view/Nightly/job/nightly-android/).

Also available are precompiled native libraries, and a jcenter package that you
may simply drop into your own applications.

## Running the Demo

Once the app is installed it can be started via the "AI Assit" icon.

While running the activities, pressing the volume keys on your device will
toggle debug visualizations on/off, rendering additional info to the screen that
may be useful for development purposes.

## Building in Android Studio using the TensorFlow AAR from JCenter

The simplest way to compile the demo app yourself, and try out changes to the
project code is to use AndroidStudio. Simply set this `android` directory as the
project root.

Then edit the `build.gradle` file and change the value of `nativeBuildSystem` to
`'none'` so that the project is built in the simplest way possible:

```None
def nativeBuildSystem = 'none'
```

While this project includes full build integration for TensorFlow, this setting
disables it, and uses the TensorFlow Inference Interface package from JCenter.

Note: Currently, in this build mode, YUV -> RGB is done using a less efficient
Java implementation, and object tracking is not available in the "TF Detect"
activity. Setting the build system to `'cmake'` currently only builds
`libtensorflow_demo.so`, which provides fast YUV -> RGB conversion and object
tracking, while still acquiring TensorFlow support via the downloaded AAR, so it
may be a lightweight way to enable these features.

For any project that does not include custom low level TensorFlow code, this is
likely sufficient.

### Bazel

NOTE: Bazel does not currently support building for Android on Windows. Full
support for gradle/cmake builds is coming soon, but in the meantime we suggest
that Windows users download the [prebuilt
binaries](https://ci.tensorflow.org/view/Nightly/job/nightly-android/) instead.

##### Install Bazel and Android Prerequisites

Bazel is the primary build system for TensorFlow. To build with Bazel, it and
the Android NDK and SDK must be installed on your system.

1.  Install the latest version of Bazel as per the instructions [on the Bazel
    website](https://bazel.build/versions/master/docs/install.html).
2.  The Android NDK is required to build the native (C/C++) TensorFlow code. The
    current recommended version is 14b, which may be found
    [here](https://developer.android.com/ndk/downloads/older_releases.html#ndk-14b-downloads).
3.  The Android SDK and build tools may be obtained
    [here](https://developer.android.com/tools/revisions/build-tools.html), or
    alternatively as part of [Android
    Studio](https://developer.android.com/studio/index.html). Build tools API >=
    23 is required to build the TF Android demo (though it will run on API >= 21
    devices).

##### Edit WORKSPACE

The Android entries in
[`<workspace_root>/WORKSPACE`](./WORKSPACE) must be uncommented
with the paths filled in appropriately depending on where you installed the NDK
and SDK. Otherwise an error such as: "The external label
'//external:android/sdk' is not bound to anything" will be reported.

Also edit the API levels for the SDK in WORKSPACE to the highest level you have
installed in your SDK. This must be >= 23 (this is completely independent of the
API level of the demo, which is defined in AndroidManifest.xml). The NDK API
level may remain at 14.

##### Install Model Files (optional)

The TensorFlow `GraphDef`s that contain the model definitions and weights are
not packaged in the repo because of their size. They are downloaded
automatically and packaged with the APK by Bazel via a new_http_archive defined
in `WORKSPACE` during the build process, and by Gradle via
download-models.gradle.

**Optional**: If you wish to place the models in your assets manually, remove
all of the `model_files` entries from the `assets` list in `avengers-android-AI-Assist`
found in the `[BUILD](BUILD)` file. Then download and extract the archives
yourself to the `assets` directory in the source tree:

```bash
BASE_URL=https://storage.googleapis.com/download.tensorflow.org/models
for MODEL_ZIP in inception5h.zip ssd_mobilenet_v1_android_export.zip stylize_v1.zip
do
  curl -L ${BASE_URL}/${MODEL_ZIP} -o /tmp/${MODEL_ZIP}
  unzip /tmp/${MODEL_ZIP} -d assets/
done
```

This will extract the models and their associated metadata files to the local
assets/ directory.

If you are using Gradle, make sure to remove download-models.gradle reference
from build.gradle after your manually download models; otherwise gradle might
download models again and overwrite your models.

##### Build

After editing your WORKSPACE file to update the SDK/NDK configuration, you may
build the APK. Run this from your workspace root:

```bash
bazel build -c opt ai_assist.apk
```

##### Install

Make sure that adb debugging is enabled on your Android 5.0 (API 21) or later
device, then after building use the following command from your workspace root
to install the APK:

```bash
adb install -r bazel-bin/ai_assist.apk
```

### Android Studio with Bazel

Android Studio may be used to build the demo in conjunction with Bazel. First,
make sure that you can build with Bazel following the above directions. Then,
look at [build.gradle](build.gradle) and make sure that the path to Bazel
matches that of your system.

At this point you can add the tensorflow/examples/android directory as a new
Android Studio project. Click through installing all the Gradle extensions it
requests, and you should be able to have Android Studio build the demo like any
other application (it will call out to Bazel to build the native code with the
NDK).

