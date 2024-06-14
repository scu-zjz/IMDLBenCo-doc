export const redirects = JSON.parse("{}")

export const routes = Object.fromEntries([
  ["/get-started.html", { loader: () => import(/* webpackChunkName: "get-started.html" */"D:/workspace/IMDLBenCo-doc/vuepress-starter/docs/.vuepress/.temp/pages/get-started.html.js"), meta: {"title":"Get Started"} }],
  ["/", { loader: () => import(/* webpackChunkName: "index.html" */"D:/workspace/IMDLBenCo-doc/vuepress-starter/docs/.vuepress/.temp/pages/index.html.js"), meta: {"title":"Home"} }],
  ["/zh/get-started.html", { loader: () => import(/* webpackChunkName: "zh_get-started.html" */"D:/workspace/IMDLBenCo-doc/vuepress-starter/docs/.vuepress/.temp/pages/zh/get-started.html.js"), meta: {"title":"快速上手"} }],
  ["/zh/", { loader: () => import(/* webpackChunkName: "zh_index.html" */"D:/workspace/IMDLBenCo-doc/vuepress-starter/docs/.vuepress/.temp/pages/zh/index.html.js"), meta: {"title":"Home"} }],
  ["/404.html", { loader: () => import(/* webpackChunkName: "404.html" */"D:/workspace/IMDLBenCo-doc/vuepress-starter/docs/.vuepress/.temp/pages/404.html.js"), meta: {"title":""} }],
]);

if (import.meta.webpackHot) {
  import.meta.webpackHot.accept()
  if (__VUE_HMR_RUNTIME__.updateRoutes) {
    __VUE_HMR_RUNTIME__.updateRoutes(routes)
  }
  if (__VUE_HMR_RUNTIME__.updateRedirects) {
    __VUE_HMR_RUNTIME__.updateRedirects(redirects)
  }
}

if (import.meta.hot) {
  import.meta.hot.accept(({ routes, redirects }) => {
    __VUE_HMR_RUNTIME__.updateRoutes(routes)
    __VUE_HMR_RUNTIME__.updateRedirects(redirects)
  })
}
