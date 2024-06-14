export const siteData = JSON.parse("{\"base\":\"/IMDLBenCo-doc/\",\"lang\":\"en-US\",\"title\":\"\",\"description\":\"\",\"head\":[],\"locales\":{\"/\":{\"lang\":\"en-US\",\"title\":\"IMDLBenCo Documentation\",\"description\":\"Benchmark and Codebase for Image manipulation localization & detection\"},\"/zh/\":{\"lang\":\"zh-CN\",\"title\":\"IMDLBenCo 文档\",\"description\":\"图像篡改检测与定位基准代码库\"}}}")

if (import.meta.webpackHot) {
  import.meta.webpackHot.accept()
  if (__VUE_HMR_RUNTIME__.updateSiteData) {
    __VUE_HMR_RUNTIME__.updateSiteData(siteData)
  }
}

if (import.meta.hot) {
  import.meta.hot.accept(({ siteData }) => {
    __VUE_HMR_RUNTIME__.updateSiteData(siteData)
  })
}
