import type { NavbarConfig } from '@vuepress/theme-default'
import { version } from '../meta.js'


export const navbarZh: NavbarConfig = [
    {
        text: '指南',
        children: [
            {
                text: '基本信息',
                children: [
                    "/zh/guide/base/introduction.md",
                    "/zh/guide/base/framework.md",
                ],
            },
            {
                text: '快速上手',
                children: [
                    "/zh/guide/quickstart/install.md",
                    "/zh/guide/quickstart/0_dataprepare.md",
                    "/zh/guide/quickstart/2_model_zoo.md",
                    "/zh/guide/quickstart/3_load_ckpt.md",
                    "/zh/guide/quickstart/1_demo.md"
                    
                ],
            },
        ],
    },

    {
        text: 'API文档',
        children: [
            "/zh/API/intro.md",
        ],
    },

    {
        text: '模型与模块',


        children: [
            {
                text: 'Model zoo',
                link: '/zh/model/model_zoo/intro&content.md',
                children: [
                    // '/zh/model/model_zoo/intro&content.md',
                    '/zh/model/model_zoo/leaderboard.md',

                ],
            },
            {
                text: 'Backbone models',
                link: '/zh/model/backbone/intro&content.md',
                children: [
                    // '/zh/model/backbone/intro&content.md',
                ],
            },
            {
                text: 'Extractor modules',
                link: '/zh/model/extractor/intro&content.md',
                children: [
                    // '/zh/model/extractor/intro&content.md',
                    // '/zh/reference/bundler/webpack.md',
                ],
            },
        ],
    },

]