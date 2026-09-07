package com.margelo.nitro.nitrospeakerid
  
import com.facebook.proguard.annotations.DoNotStrip

@DoNotStrip
class NitroSpeakerId : HybridNitroSpeakerIdSpec() {
  override fun multiply(a: Double, b: Double): Double {
    return a * b
  }
}
