import{_ as o,c as d,a as t,e as r,w as s,r as p,o as m,b as e,d as n}from"./app-DCerHgAi.js";const h={};function u(b,a){const c=p("Tabs");return m(),d("div",null,[a[4]||(a[4]=t(`<h1 id="安装" tabindex="-1"><a class="header-anchor" href="#安装"><span>安装</span></a></h1><p><strong>请注意，与大多数Paper提供的代码不同，Benco的仓库本身并不期望通过git clone等方式完成使用，因为实际代码中涉及到大量的工程开发所需的组件。期望的使用方法是直接通过pip install并将其当做Python库使用</strong></p><h2 id="对于普通用户" tabindex="-1"><a class="header-anchor" href="#对于普通用户"><span>对于普通用户</span></a></h2><p>如果你只希望用IMDL-BenCo复现论文，并构建自己的模型，则IMDL-BenCo的安装方式非常简单，目前通过PyPI进行了包管理，直接通过如下指令即可完成</p><div class="language-bash line-numbers-mode" data-highlighter="prismjs" data-ext="sh"><pre><code><span class="line">pip <span class="token function">install</span> imdlbenco</span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div></div></div><p>如果安装成功，在命令行中运行如下指令可以验证是否安装成功，并且自动检查是否有新版本。（本仓库处于迭代开发中，最好经常检查版本更新）</p>`,6)),r(c,{id:"16",data:[{id:"简写命令"},{id:"完整命令"}]},{title0:s(({value:i,isActive:l})=>a[0]||(a[0]=[n("简写命令")])),title1:s(({value:i,isActive:l})=>a[1]||(a[1]=[n("完整命令")])),tab0:s(({value:i,isActive:l})=>a[2]||(a[2]=[e("div",{class:"language-bash line-numbers-mode","data-highlighter":"prismjs","data-ext":"sh"},[e("pre",null,[e("code",null,[e("span",{class:"line"},[n("benco "),e("span",{class:"token parameter variable"},"-v"),n()]),n(`
`),e("span",{class:"line"})])]),e("div",{class:"line-numbers","aria-hidden":"true",style:{"counter-reset":"line-number 0"}},[e("div",{class:"line-number"})])],-1)])),tab1:s(({value:i,isActive:l})=>a[3]||(a[3]=[e("div",{class:"language-bash line-numbers-mode","data-highlighter":"prismjs","data-ext":"sh"},[e("pre",null,[e("code",null,[e("span",{class:"line"},[n("benco "),e("span",{class:"token parameter variable"},"--version")]),n(`
`),e("span",{class:"line"})])]),e("div",{class:"line-numbers","aria-hidden":"true",style:{"counter-reset":"line-number 0"}},[e("div",{class:"line-number"})])],-1)])),_:1}),a[5]||(a[5]=t(`<p>如果正常安装了最新版本，应该可以看到如下内容：</p><div class="language-text line-numbers-mode" data-highlighter="prismjs" data-ext="text"><pre><code><span class="line">IMDLBenCo codebase version: 0.1.21</span>
<span class="line">        Checking for updates...</span>
<span class="line">        Local version:  0.1.21</span>
<span class="line">        PyPI newest version:  0.1.21</span>
<span class="line">You are using the latest version: 0.1.21.</span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>像其他的命令行工具一样，运行<code>benco -h</code>可以查看帮助引导等内容。</p><h2 id="对于想为官方仓库贡献的开发者" tabindex="-1"><a class="header-anchor" href="#对于想为官方仓库贡献的开发者"><span>对于想为官方仓库贡献的开发者</span></a></h2><p>如果你试图为IMDL-BenCo的<strong>Python Library</strong>在开发新功能并贡献到官方仓库，则需要按照本段完成。推荐你先在环境中卸载所有已经安装的IMDL-BenCo，然后克隆您fork过的IMDL-BenCo的仓库后，切换到dev分支获得最新的“开发版本”后，并使用特殊的<code>pip install -e . </code>指令完成本地安装，这会使得当前Python环境始终根据本路径下的包所含的脚本执行IMDL-BenCo库，并在更新文件时自动更新相应执行行为，非常便于调试开发。</p><div class="language-bash line-numbers-mode" data-highlighter="prismjs" data-ext="sh"><pre><code><span class="line"><span class="token comment"># 卸载已有的 IMDL-BenCo 库</span></span>
<span class="line">pip uninstall imdlbenco</span>
<span class="line"></span>
<span class="line"><span class="token comment"># 克隆 GitHub 上的 IMDL-BenCo 的fork过的仓库</span></span>
<span class="line"><span class="token function">git</span> clone https://github.com/your_name/IMDL-BenCo.git</span>
<span class="line"></span>
<span class="line"><span class="token comment"># 进入项目目录</span></span>
<span class="line"><span class="token builtin class-name">cd</span> IMDL-BenCo</span>
<span class="line"></span>
<span class="line"><span class="token comment"># 使用 \`pip install -e .\` 进行本地开发安装</span></span>
<span class="line">pip <span class="token function">install</span> <span class="token parameter variable">-e</span> <span class="token builtin class-name">.</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 验证安装</span></span>
<span class="line">pip show imdlbenco</span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>如果安装正常，执行<code>pip list</code>后应当看到</p><div class="language-text line-numbers-mode" data-highlighter="prismjs" data-ext="text"><pre><code><span class="line">Package                 Version            Editable project location</span>
<span class="line">----------------------- ------------------ ------------------------------------------------------</span>
<span class="line">...</span>
<span class="line">IMDLBenCo               0.1.10             /mnt/data0/xiaochen/workspace/IMDLBenCo_pure/IMDLBenCo</span>
<span class="line">...</span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p><code>Editable Project Location</code> 这一栏有对应的路径，即代表所有对于该路径下的python脚本修改可以直接生效于该Python环境内部，无需重新安装，非常便于调试。</p>`,9))])}const g=o(h,[["render",u]]),f=JSON.parse('{"path":"/zh/guide/quickstart/install.html","title":"安装","lang":"zh-CN","frontmatter":{"description":"安装 请注意，与大多数Paper提供的代码不同，Benco的仓库本身并不期望通过git clone等方式完成使用，因为实际代码中涉及到大量的工程开发所需的组件。期望的使用方法是直接通过pip install并将其当做Python库使用 对于普通用户 如果你只希望用IMDL-BenCo复现论文，并构建自己的模型，则IMDL-BenCo的安装方式非常简单，...","head":[["link",{"rel":"alternate","hreflang":"en-us","href":"https://scu-zjz.github.io/IMDLBenCo-doc/IMDLBenCo-doc/guide/quickstart/install.html"}],["meta",{"property":"og:url","content":"https://scu-zjz.github.io/IMDLBenCo-doc/IMDLBenCo-doc/zh/guide/quickstart/install.html"}],["meta",{"property":"og:site_name","content":"IMDLBenCo 文档"}],["meta",{"property":"og:title","content":"安装"}],["meta",{"property":"og:description","content":"安装 请注意，与大多数Paper提供的代码不同，Benco的仓库本身并不期望通过git clone等方式完成使用，因为实际代码中涉及到大量的工程开发所需的组件。期望的使用方法是直接通过pip install并将其当做Python库使用 对于普通用户 如果你只希望用IMDL-BenCo复现论文，并构建自己的模型，则IMDL-BenCo的安装方式非常简单，..."}],["meta",{"property":"og:type","content":"article"}],["meta",{"property":"og:locale","content":"zh-CN"}],["meta",{"property":"og:locale:alternate","content":"en-US"}],["meta",{"property":"og:updated_time","content":"2025-03-30T13:14:09.000Z"}],["meta",{"property":"article:modified_time","content":"2025-03-30T13:14:09.000Z"}],["script",{"type":"application/ld+json"},"{\\"@context\\":\\"https://schema.org\\",\\"@type\\":\\"Article\\",\\"headline\\":\\"安装\\",\\"image\\":[\\"\\"],\\"dateModified\\":\\"2025-03-30T13:14:09.000Z\\",\\"author\\":[]}"]]},"headers":[{"level":2,"title":"对于普通用户","slug":"对于普通用户","link":"#对于普通用户","children":[]},{"level":2,"title":"对于想为官方仓库贡献的开发者","slug":"对于想为官方仓库贡献的开发者","link":"#对于想为官方仓库贡献的开发者","children":[]}],"git":{"updatedTime":1743340449000,"contributors":[{"name":"Ma Xiaochen (马晓晨)","username":"","email":"mxch1122@126.com","commits":9},{"name":"Ma, Xiaochen","username":"","email":"mxch1122@126.com","commits":1}],"changelog":[{"hash":"e7e94b639b491b3bf1b5f0d88e23bc7b0135d8f1","time":1743340449000,"email":"mxch1122@126.com","author":"Ma Xiaochen (马晓晨)","message":"[update] add english version for demo. revise readme. remove redundant line in install"},{"hash":"5398ea0cf41847f4d5f5fd0ebbe3bcbb5dec7114","time":1743339971000,"email":"mxch1122@126.com","author":"Ma Xiaochen (马晓晨)","message":"[update] update version and modify benco -v"},{"hash":"6db2d2f20cb8ff2505eff176cbe1d5c02670f23e","time":1737960496000,"email":"mxch1122@126.com","author":"Ma Xiaochen (马晓晨)","message":"[update] add benco --version check"},{"hash":"6eef1ddbce0eb8c8c567eb00ea32854db737644c","time":1729058517000,"email":"mxch1122@126.com","author":"Ma, Xiaochen","message":"Update install.md"},{"hash":"dd8ba6220fff989633869274cd8b32fe443434e7","time":1729058253000,"email":"mxch1122@126.com","author":"Ma Xiaochen (马晓晨)","message":"[update] revise phrasing in install.md"},{"hash":"8c26dc30c467afbc8ec70ff807ceaca6c1a0e595","time":1725040789000,"email":"mxch1122@126.com","author":"Ma Xiaochen (马晓晨)","message":"[update] add document for model_zoo"},{"hash":"6de34ba84e800741a8134cad3c0a4c9925e096ef","time":1724932687000,"email":"mxch1122@126.com","author":"Ma Xiaochen (马晓晨)","message":"[update] add guide and dataprepare process"},{"hash":"a8f6cb56d3483ba336ed64a05ab07ab637eb4db7","time":1719873732000,"email":"mxch1122@126.com","author":"Ma Xiaochen (马晓晨)","message":"[update] add install for developer"},{"hash":"c71bf1475c5477e026be4d4a8bb970dcb239b05e","time":1719867818000,"email":"mxch1122@126.com","author":"Ma Xiaochen (马晓晨)","message":"[update] add inrtro to guide, en &amp; zh"},{"hash":"ac6672446ff126bc72155a79ec8a98f6591aacaa","time":1719246628000,"email":"mxch1122@126.com","author":"Ma Xiaochen (马晓晨)","message":"[update] add navbar sidebar to Chinese version"}]},"filePathRelative":"zh/guide/quickstart/install.md","autoDesc":true}');export{g as comp,f as data};
