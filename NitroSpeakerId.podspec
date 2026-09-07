require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

Pod::Spec.new do |s|
  s.name         = "NitroSpeakerId"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.homepage     = package["homepage"]
  s.license      = package["license"]
  s.authors      = package["author"]

  s.platforms    = { :ios => min_ios_version_supported }
  s.source       = { :git => "https://github.com/Gautham495/react-native-nitro-speaker-id.git", :tag => "#{s.version}" }

  s.source_files = [
    "ios/**/*.{swift,h,m,mm,cpp,hpp}",
    "nitrogen/generated/ios/**/*.{swift,h,m,mm,cpp,hpp}"
  ]

  s.pod_target_xcconfig = {
    "DEFINES_MODULE" => "YES",
    "EXCLUDED_ARCHS[sdk=iphonesimulator*]" => "x86_64"
  }

  # Core ML is a system framework — already on every iOS device, no pod
  # dependency, no MB added to your app binary. This is the whole point
  # of using platform-native runtimes over ONNX Runtime: zero third-party
  # footprint, zero version drama, zero 16 KB alignment fights.
  s.frameworks = ["CoreML", "Foundation"]

  load "nitrogen/generated/ios/NitroSpeakerId+autolinking.rb"
  add_nitrogen_files(s)

  s.dependency "React-jsi"
  s.dependency "React-callinvoker"

  install_modules_dependencies(s)
end