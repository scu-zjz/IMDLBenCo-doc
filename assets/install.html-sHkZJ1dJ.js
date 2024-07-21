import{_ as n,c as s,o as a,d as e}from"./app-iKWZIQs7.js";const l={},i=e(`<h1 id="安装" tabindex="-1"><a class="header-anchor" href="#安装"><span>安装</span></a></h1><h2 id="普通用户" tabindex="-1"><a class="header-anchor" href="#普通用户"><span>普通用户</span></a></h2><p>IMDL-BenCo的安装方式非常简单，目前通过PyPI进行了包管理，直接通过如下指令即可完成</p><div class="language-bash line-numbers-mode" data-highlighter="prismjs" data-ext="sh" data-title="sh"><pre class="language-bash"><code><span class="line">pip <span class="token function">install</span> imdlbenco</span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div></div></div><h2 id="开发者" tabindex="-1"><a class="header-anchor" href="#开发者"><span>开发者</span></a></h2><p>如果你试图为IMDL-BenCo在本地开发新功能，推荐你先在环境中卸载所有已经安装的IMDL-BenCo，然后克隆您fork过的IMDL-BenCo的仓库后，切换到dev分支获得最新的“开发版本”后，并使用特殊的<code>pip install -e . </code>指令完成本地安装，这会使得当前Python环境始终根据本路径下的包所含的脚本执行IMDL-BenCo库的行文，并在更新文件时自动更新相应行为，非常便于调试开发。</p><div class="language-bash line-numbers-mode" data-highlighter="prismjs" data-ext="sh" data-title="sh"><pre class="language-bash"><code><span class="line"><span class="token comment"># 卸载已有的 IMDL-BenCo 库</span></span>
<span class="line">pip uninstall imdlbenco</span>
<span class="line"></span>
<span class="line"><span class="token comment"># 克隆 GitHub 上的 IMDL-BenCo 的fork过的仓库</span></span>
<span class="line"><span class="token function">git</span> clone https://github.com/your_name/IMDL-BenCo.git</span>
<span class="line"></span>
<span class="line"><span class="token comment"># 进入项目目录</span></span>
<span class="line"><span class="token builtin class-name">cd</span> IMDL-BenCo</span>
<span class="line"></span>
<span class="line"><span class="token comment"># 切换到 dev 分支</span></span>
<span class="line"><span class="token function">git</span> checkout dev</span>
<span class="line"></span>
<span class="line"><span class="token comment"># 使用 \`pip install -e .\` 进行本地开发安装</span></span>
<span class="line">pip <span class="token function">install</span> <span class="token parameter variable">-e</span> <span class="token builtin class-name">.</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 验证安装</span></span>
<span class="line">pip show imdl-benco</span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>`,7),c=[i];function t(p,d){return a(),s("div",null,c)}const r=n(l,[["render",t],["__file","install.html.vue"]]),m=JSON.parse('{"path":"/zh/guide/quickstart/install.html","title":"安装","lang":"zh-CN","frontmatter":{},"headers":[{"level":2,"title":"普通用户","slug":"普通用户","link":"#普通用户","children":[]},{"level":2,"title":"开发者","slug":"开发者","link":"#开发者","children":[]}],"git":{"updatedTime":1719873732000,"contributors":[{"name":"Ma Xiaochen (马晓晨)","email":"mxch1122@126.com","commits":3}]},"filePathRelative":"zh/guide/quickstart/install.md"}');export{r as comp,m as data};
