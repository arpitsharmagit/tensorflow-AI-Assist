package org.tensorflow.demo;

/**
 * Created by arpit.sharma1 on 10/27/2017.
 */


import android.media.AudioManager;
import android.os.Build;
import android.speech.RecognizerIntent;
import android.speech.tts.TextToSpeech;
import android.speech.tts.TextToSpeech.OnInitListener;
import android.speech.tts.UtteranceProgressListener;
import android.util.Log;

import java.util.HashMap;
import android.content.Context;
import java.util.Locale;

public class Speaker implements OnInitListener {

    private TextToSpeech tts;

    private boolean ready = false;

    private boolean allowed = false;

    public Speaker(Context context){
        tts = new TextToSpeech(context, this);
    }

    @Override
    public void onInit(int status) {
        if(status == TextToSpeech.SUCCESS){
            tts.setLanguage(Locale.US);
            ready = true;
        }else{
            ready = false;
        }
    }
    public boolean isAllowed(){
        return allowed;
    }

    public void allow(boolean allowed){
        this.allowed = allowed;
    }

    public void speak(String text){
        if(ready && allowed) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                tts.speak(text, TextToSpeech.QUEUE_ADD, null, null);
            }else{
                tts.speak(text, TextToSpeech.QUEUE_ADD, null);
            }
        }
    }
    public void pause(int duration){
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            tts.playSilentUtterance(duration,TextToSpeech.QUEUE_ADD,null);
        }
        else{
            tts.playSilence(duration, TextToSpeech.QUEUE_ADD, null);
        }
    }
    // Free up resources
    public void destroy(){
        tts.shutdown();
    }
}


