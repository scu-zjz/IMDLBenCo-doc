import type { NavbarConfig } from '@vuepress/theme-default'
import { version } from '../meta.js'

export const navbarEn: NavbarConfig = [
    {
        text: 'Guide',
        children: [
            {
                text: 'Basic Information',
                children: [
                    "/guide/base/introduction.md",
                    "/guide/base/framework.md",
                ],
            },
            {
                text: 'Quick Start',
                children: [
                    "/guide/quickstart/install.md",
                    "/guide/quickstart/0_dataprepare.md",
                    "/guide/quickstart/1_model_zoo.md",
                    "/guide/quickstart/2_load_ckpt.md",
                    "/guide/quickstart/3_demo.md"
                    
                ],
            },
        ],
    },

    {
        text: 'API Reference',
        children: [
            "/API/intro.md",
        ],
    },
    {
        text: 'IMDL data & model hub',

        children: [
            {
                text: 'Datasets',
                children: [
                    "/imdl_data_model_hub/data/IMDLdatasets.md",
                    "/imdl_data_model_hub/data/AIGCdatasets.md"
                ],
            },
            {
                text: 'Models & Papers',
                children: [
                    "/imdl_data_model_hub/models/benco.md",
                    "/imdl_data_model_hub/models/general.md"
                ],
            }
        ]
    }
    // {
    //     text: 'Models & modules',


    //     children: [
    //         {
    //             text: 'Model zoo',
    //             link: '/model/model_zoo/intro&content.md',
    //             children: [
    //                 // '/zh/model/model_zoo/intro&content.md',
    //                 '/model/model_zoo/leaderboard.md',

    //             ],
    //         },
    //         {
    //             text: 'Backbone models',
    //             link: '/model/backbone/intro&content.md',
    //             children: [
    //                 // '/zh/model/backbone/intro&content.md',
    //             ],
    //         },
    //         {
    //             text: 'Extractor modules',
    //             link: '/model/extractor/intro&content.md',
    //             children: [
    //                 // '/zh/model/extractor/intro&content.md',
    //                 // '/zh/reference/bundler/webpack.md',
    //             ],
    //         },
    //     ],
    // },

]