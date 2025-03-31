import { defaultTheme } from '@vuepress/theme-default'
import { defineUserConfig } from 'vuepress/cli'
import { viteBundler } from '@vuepress/bundler-vite'
import { searchPlugin } from '@vuepress/plugin-search'
import { docsearchPlugin } from '@vuepress/plugin-docsearch'

import {
  head,
  navbarEn,
  navbarZh,
  sidebarEn,
  sidebarZh,
} from './configs/index.js'

export default defineUserConfig({
  base: '/IMDLBenCo-doc/',

  head,

  locales: {
    '/': {
      lang: 'en-US',
      title: 'IMDLBenCo Documentation',
      description: 'Benchmark and Codebase for Image manipulation localization & detection',
    },
    // '/en/': {
    //   lang: 'en-US',
    //   title: 'IMDLBenCo Documentation',
    //   description: 'Benchmark and Codebase for Image manipulation localization & detection',
    // },
    '/zh/': {
      lang: 'zh-CN',
      title: 'IMDLBenCo 文档',
      description: '图像篡改检测与定位基准代码库',
    },
  },

  theme: defaultTheme({
    logo: '/images/IMDL_BenCo.png',
    repo: 'scu-zjz/IMDLBenCo',
    docsRepo: 'scu-zjz/IMDLBenCo-doc',
    // docsRepo: 'vuepress/docs',
    docsDir: 'docs',


    locales: {
      /**
       * English locale config
       *
       * As the default locale of @vuepress/theme-default is English,
       * we don't need to set all of the locale fields
       */

      '/': {
        // navbar
        navbar: navbarEn,
        selectLanguageName: 'English',
        // sidebar
        sidebar: sidebarEn,
        // page meta
        editLinkText: 'Edit this page on GitHub',
      },

      // '/en/': {
      //   // navbar
      //   navbar: navbarEn,
      //   selectLanguageName: 'English',
      //   // sidebar
      //   sidebar: sidebarEn,
      //   // page meta
      //   editLinkText: 'Edit this page on GitHub',
      // },

      /**
       * Chinese locale config
       */
      '/zh/': {
        // navbar
        navbar: navbarZh,
        selectLanguageName: '简体中文',
        selectLanguageText: '选择语言',
        selectLanguageAriaLabel: '选择语言',
        // sidebar
        sidebar: sidebarZh,
        // page meta
        editLinkText: '在 GitHub 上编辑此页',
        lastUpdatedText: '上次更新',
        contributorsText: '贡献者',
        // custom containers
        tip: '提示',
        warning: '注意',
        danger: '警告',
        // 404 page
        notFound: [
          '这里什么都没有',
          '我们怎么到这来了？',
          '这是一个 404 页面',
          '看起来我们进入了错误的链接',
        ],
        backToHome: '返回首页',
        // a11y
        openInNewWindow: '在新窗口打开',
        toggleColorMode: '切换颜色模式',
        toggleSidebar: '切换侧边栏',
      },
    }
  }),

  bundler: viteBundler(),
  plugins:[
    // 一个本地的普通搜索，后续可以考虑换成docsearch
    // https://ecosystem.vuejs.press/zh/plugins/search/search.html#%E4%BD%BF%E7%94%A8%E6%96%B9%E6%B3%95
    // 以下是docsearch的官方网站和文档
    // https://ecosystem.vuejs.press/zh/plugins/search/docsearch.html
    // https://docsearch.algolia.com/docs/tips/
    // https://www.algolia.com/developers?utm_source=vuepress.vuejs.org&utm_medium=referral&utm_content=powered_by&utm_campaign=docsearch
    searchPlugin({
        locales: {
          '/en/': {
            placeholder: 'Search',
          },
          '/zh': {
            placeholder: '搜索',
          },
        },
        // 最大搜索结果数
        maxSuggestions: 10,
        // 需要搜索的字段
        searchMaxSuggestions: 10,
        hotKeys: ['s', '/'],
        // 排除标题中的某些词语
        // isSearchable: (page) => page.path !== '/exclude.html',
    }),
    docsearchPlugin({
      // 配置项
      apiKey : "9d2d7b0a01f010b60f525b7a4a74b841",
      appId : "W3JD3JHPAP",
      indexName: "scu-zjzio",
      insights: true,
      container: '#algolia-doc-search',
      debug: false,
    }),
  ]
  // Redirection logic
})

