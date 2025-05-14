import type { SidebarConfig } from '@vuepress/theme-default'

export const sidebarZh: SidebarConfig = {
    '/zh/guide/': [
        {
            text: '指南',
            children: [
                '/zh/guide/base/introduction.md',
                '/zh/guide/base/framework.md',
                // '/zh/guide/getting-started.md',
                // '/zh/guide/configuration.md',
                // '/zh/guide/page.md',
                // '/zh/guide/markdown.md',
                // '/zh/guide/assets.md',
                // '/zh/guide/i18n.md',
                // '/zh/guide/deployment.md',
                // '/zh/guide/theme.md',
                // '/zh/guide/plugin.md',
                // '/zh/guide/bundler.md',
                // '/zh/guide/migration.md',
                // '/zh/guide/troubleshooting.md',
            ],
        },
        {
            text: '快速上手',
            children: [
                '/zh/guide/quickstart/install.md',
                "/zh/guide/quickstart/0_dataprepare.md",
                "/zh/guide/quickstart/1_model_zoo.md",
                "/zh/guide/quickstart/2_load_ckpt.md",
                "/zh/guide/quickstart/3_demo.md",
                "/zh/guide/quickstart/4_save_images.md"
                // '/zh/guide/getting-started.md',
                // '/zh/guide/configuration.md',
                // '/zh/guide/page.md',
                // '/zh/guide/markdown.md',
                // '/zh/guide/assets.md',
                // '/zh/guide/i18n.md',
                // '/zh/guide/deployment.md',
                // '/zh/guide/theme.md',
                // '/zh/guide/plugin.md',
                // '/zh/guide/bundler.md',
                // '/zh/guide/migration.md',
                // '/zh/guide/troubleshooting.md',
            ],
        },
    ],
    '/zh/API/': [
        {
            text: '数据相关',
            children: [
                '/zh/advanced/architecture.md',
                '/zh/advanced/plugin.md',
                '/zh/advanced/theme.md',
            ],
        },
        {
            text: 'Cookbook',
            children: [
                '/zh/advanced/cookbook/README.md',
                '/zh/advanced/cookbook/usage-of-client-config.md',
                '/zh/advanced/cookbook/adding-extra-pages.md',
                '/zh/advanced/cookbook/making-a-theme-extendable.md',
                '/zh/advanced/cookbook/passing-data-to-client-code.md',
                '/zh/advanced/cookbook/markdown-and-vue-sfc.md',
                '/zh/advanced/cookbook/resolving-routes.md',
            ],
        },
    ],
    '/zh/model/': [
        {
            text: 'Model zoo',
            collapsible: true,
            children: [
                '/zh/model/model_zoo/intro&content.md',
                '/zh/model/model_zoo/mvss.md',
                '/zh/model/model_zoo/trufor.md',
                '/zh/model/model_zoo/iml_vit.md'
            ],
        },
        {
            text: '打包工具',
            children: [
                '/zh/reference/bundler/vite.md',
                '/zh/reference/bundler/webpack.md',
            ],
        },
        {
            text: '生态系统',
            children: [
                {
                    text: '默认主题',
                    link: 'https://ecosystem.vuejs.press/zh/themes/default/',
                },
                {
                    text: '插件',
                    link: 'https://ecosystem.vuejs.press/zh/plugins/',
                },
            ],
        },
    ],
    '/zh/imdl_data_model_hub/': [
        {
            text: '篡改检测信息集合',
            children: [
                {
                    text: '数据集',
                    children: [
                        "/zh/imdl_data_model_hub/data/IMDLdatasets.md",
                        "/zh/imdl_data_model_hub/data/AIGCdatasets.md"
                    ],
                },
                {
                    text: '模型和论文',
                    children: [
                        "/zh/imdl_data_model_hub/models/benco.md",
                        "/zh/imdl_data_model_hub/models/general.md"
                    ],
                }
            ],
        },
    ]
}