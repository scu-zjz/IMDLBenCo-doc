import type { SidebarConfig } from '@vuepress/theme-default'

export const sidebarEn: SidebarConfig = {
    '/en/guide/': [
        {
            text: 'Guide',
            children: [
                '/en/guide/base/introduction.md',
                // '/en/guide/getting-started.md',
                // '/en/guide/configuration.md',
                // '/en/guide/page.md',
                // '/en/guide/markdown.md',
                // '/en/guide/assets.md',
                // '/en/guide/i18n.md',
                // '/en/guide/deployment.md',
                // '/en/guide/theme.md',
                // '/en/guide/plugin.md',
                // '/en/guide/bundler.md',
                // '/en/guide/migration.md',
                // '/en/guide/troubleshooting.md',
            ],
        },
        {
            text: 'Quick Start',
            children: [
                '/en/guide/quickstart/install.md',
                // '/en/guide/getting-started.md',
                // '/en/guide/configuration.md',
                // '/en/guide/page.md',
                // '/en/guide/markdown.md',
                // '/en/guide/assets.md',
                // '/en/guide/i18n.md',
                // '/en/guide/deployment.md',
                // '/en/guide/theme.md',
                // '/en/guide/plugin.md',
                // '/en/guide/bundler.md',
                // '/en/guide/migration.md',
                // '/en/guide/troubleshooting.md',
            ],
        },
    ],
    '/en/API/': [
        {
            text: 'Data Related',
            children: [
                '/en/advanced/architecture.md',
                '/en/advanced/plugin.md',
                '/en/advanced/theme.md',
            ],
        },
        {
            text: 'Cookbook',
            children: [
                '/en/advanced/cookbook/README.md',
                '/en/advanced/cookbook/usage-of-client-config.md',
                '/en/advanced/cookbook/adding-extra-pages.md',
                '/en/advanced/cookbook/making-a-theme-extendable.md',
                '/en/advanced/cookbook/passing-data-to-client-code.md',
                '/en/advanced/cookbook/markdown-and-vue-sfc.md',
                '/en/advanced/cookbook/resolving-routes.md',
            ],
        },
    ],
    '/en/model/': [
        {
            text: 'Model Zoo',
            collapsible: true,
            children: [
                '/en/model/model_zoo/intro&content.md',
                '/en/model/model_zoo/mvss.md',
                '/en/model/model_zoo/trufor.md',
                '/en/model/model_zoo/iml_vit.md'
            ],
        },
        {
            text: 'Bundler Tools',
            children: [
                '/en/reference/bundler/vite.md',
                '/en/reference/bundler/webpack.md',
            ],
        },
        {
            text: 'Ecosystem',
            children: [
                {
                    text: 'Default Theme',
                    link: 'https://ecosystem.vuejs.press/en/themes/default/',
                },
                {
                    text: 'Plugins',
                    link: 'https://ecosystem.vuejs.press/en/plugins/',
                },
            ],
        },
    ],
}
