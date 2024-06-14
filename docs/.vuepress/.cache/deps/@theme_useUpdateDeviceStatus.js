import {
  useEventListener
} from "./chunk-RJBO7M6I.js";
import {
  onMounted
} from "./chunk-ICRYXTSQ.js";

// node_modules/@vuepress/theme-default/lib/client/composables/useUpdateDeviceStatus.js
import cssVariables from "D:/workspace/IMDLBenCo-doc/vuepress-starter/node_modules/@vuepress/theme-default/lib/client/styles/_variables.module.scss";
var DeviceType;
(function(DeviceType2) {
  DeviceType2["MOBILE"] = "mobile";
})(DeviceType || (DeviceType = {}));
var DeviceTypeMap = {
  [DeviceType.MOBILE]: Number.parseInt(cssVariables.mobile.replace("px", ""), 10)
};
var useUpdateDeviceStatus = (deviceType, callback) => {
  const width = DeviceTypeMap[deviceType];
  if (!Number.isInteger(width)) {
    if (__VUEPRESS_DEV__)
      throw new Error("device width must be a integer");
    return;
  }
  useEventListener("orientationchange", () => callback(width), false);
  useEventListener("resize", () => callback(width), false);
  onMounted(() => {
    callback(width);
  });
};
export {
  DeviceType,
  useUpdateDeviceStatus
};
//# sourceMappingURL=@theme_useUpdateDeviceStatus.js.map
