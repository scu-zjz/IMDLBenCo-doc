import type { SidebarConfig } from '@vuepress/theme-default'

export const sidebarEn: SidebarConfig = {
    '/guide/': [
        {
            text: 'Guide',
            children: [
                '/guide/base/introduction.md',
                '/guide/base/framework.md',
                // '/guide/configuration.md',
                // '/guide/page.md',
                // '/guide/markdown.md',
                // '/guide/assets.md',
                // '/guide/i18n.md',
                // '/guide/deployment.md',
                // '/guide/theme.md',
                // '/guide/plugin.md',
                // '/guide/bundler.md',
                // '/guide/migration.md',
                // '/guide/troubleshooting.md',
            ],
        },
        {
            text: 'Quick Start',
            children: [
                '/guide/quickstart/install.md',
                "/guide/quickstart/0_dataprepare.md",
                "/guide/quickstart/2_model_zoo.md",
                "/guide/quickstart/3_load_ckpt.md",
                "/guide/quickstart/1_demo.md",
                // '/guide/getting-started.md',
                // '/guide/configuration.md',
                // '/guide/page.md',
                // '/guide/markdown.md',
                // '/guide/assets.md',
                // '/guide/i18n.md',
                // '/guide/deployment.md',
                // '/guide/theme.md',
                // '/guide/plugin.md',
                // '/guide/bundler.md',
                // '/guide/migration.md',
                // '/guide/troubleshooting.md',
            ],
        },
    ],
    '/API/': [
        {
            text: 'Data Related',
            children: [
                '/advanced/architecture.md',
                '/advanced/plugin.md',
                '/advanced/theme.md',
            ],
        },
        {
            text: 'Cookbook',
            children: [
                '/advanced/cookbook/README.md',
                '/advanced/cookbook/usage-of-client-config.md',
                '/advanced/cookbook/adding-extra-pages.md',
                '/advanced/cookbook/making-a-theme-extendable.md',
                '/advanced/cookbook/passing-data-to-client-code.md',
                '/advanced/cookbook/markdown-and-vue-sfc.md',
                '/advanced/cookbook/resolving-routes.md',
            ],
        },
    ],
    '/model/': [
        {
            text: 'Model Zoo',
            collapsible: true,
            children: [
                '/model/model_zoo/intro&content.md',
                '/model/model_zoo/mvss.md',
                '/model/model_zoo/trufor.md',
                '/model/model_zoo/iml_vit.md'
            ],
        },
        {
            text: 'Bundler Tools',
            children: [
                '/reference/bundler/vite.md',
                '/reference/bundler/webpack.md',
            ],
        },
        {
            text: 'Ecosystem',
            children: [
                {
                    text: 'Default Theme',
                    link: 'https://ecosystem.vuejs.press/themes/default/',
                },
                {
                    text: 'Plugins',
                    link: 'https://ecosystem.vuejs.press/plugins/',
                },
            ],
        },
    ],
}
