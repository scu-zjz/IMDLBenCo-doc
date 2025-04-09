import{_ as r,c,a as o,e as d,w as s,r as p,o as m,b as e,d as a}from"./app-DCerHgAi.js";const u={};function h(v,n){const l=p("Tabs");return m(),c("div",null,[n[4]||(n[4]=o(`<h1 id="installation" tabindex="-1"><a class="header-anchor" href="#installation"><span>Installation</span></a></h1><p><strong>Please note that, unlike most code provided in papers, the Benco repository is not intended to be used via methods like <code>git clone</code>, as the code involves numerous components required for engineering development. The expected method of usage is through <code>pip install</code>, treating it as a Python library.</strong></p><h2 id="for-regular-users" tabindex="-1"><a class="header-anchor" href="#for-regular-users"><span>For Regular Users</span></a></h2><p>If you only wish to use IMDL-BenCo to reproduce the paper and build your own model, the installation process is very simple. Currently, IMDL-BenCo is managed via PyPI, and you can complete the installation by running the following command:</p><div class="language-bash line-numbers-mode" data-highlighter="prismjs" data-ext="sh"><pre><code><span class="line">pip <span class="token function">install</span> imdlbenco</span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div></div></div><p>You can run the following command to check If the python package of IMDLBenCo is installed properly and check for the latest version. (This repository is under rapid development and will release new version offten.)</p>`,6)),d(l,{id:"16",data:[{id:"Abbreviated Command"},{id:"Full Command"}]},{title0:s(({value:i,isActive:t})=>n[0]||(n[0]=[a("Abbreviated Command")])),title1:s(({value:i,isActive:t})=>n[1]||(n[1]=[a("Full Command")])),tab0:s(({value:i,isActive:t})=>n[2]||(n[2]=[e("div",{class:"language-bash line-numbers-mode","data-highlighter":"prismjs","data-ext":"sh"},[e("pre",null,[e("code",null,[e("span",{class:"line"},[a("benco "),e("span",{class:"token parameter variable"},"-v"),a("  ")]),a(`
`),e("span",{class:"line"})])]),e("div",{class:"line-numbers","aria-hidden":"true",style:{"counter-reset":"line-number 0"}},[e("div",{class:"line-number"})])],-1)])),tab1:s(({value:i,isActive:t})=>n[3]||(n[3]=[e("div",{class:"language-bash line-numbers-mode","data-highlighter":"prismjs","data-ext":"sh"},[e("pre",null,[e("code",null,[e("span",{class:"line"},[a("benco "),e("span",{class:"token parameter variable"},"--version"),a("  ")]),a(`
`),e("span",{class:"line"})])]),e("div",{class:"line-numbers","aria-hidden":"true",style:{"counter-reset":"line-number 0"}},[e("div",{class:"line-number"})])],-1)])),_:1}),n[5]||(n[5]=o(`<p>If you have installed latest version, you will see the following content:</p><div class="language-text line-numbers-mode" data-highlighter="prismjs" data-ext="text"><pre><code><span class="line">IMDLBenCo codebase version: 0.1.21</span>
<span class="line">        Checking for updates...</span>
<span class="line">        Local version:  0.1.21</span>
<span class="line">        PyPI newest version:  0.1.21</span>
<span class="line">You are using the latest version: 0.1.21.</span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>Just like other command line interface, you can run <code>benco -h</code> to check for help and guidance.</p><h2 id="for-developers-contributing-to-the-official-repository" tabindex="-1"><a class="header-anchor" href="#for-developers-contributing-to-the-official-repository"><span>For Developers Contributing to the Official Repository</span></a></h2><p>If you are trying to develop new features for the <strong>IMDL-BenCo Python Library</strong> and contribute to the official repository, follow the instructions in this section. It is recommended to first uninstall any previously installed versions of IMDL-BenCo in your environment. Then, clone your forked repository of IMDL-BenCo, switch to the <code>dev</code> branch to get the latest &quot;development version,&quot; and use the special command <code>pip install -e .</code> to complete the local installation. This will ensure that the current Python environment always executes the IMDL-BenCo library based on the scripts in this directory and automatically updates the corresponding behavior when files are updated, which is highly convenient for debugging and development.</p><div class="language-bash line-numbers-mode" data-highlighter="prismjs" data-ext="sh"><pre><code><span class="line"><span class="token comment"># Uninstall any existing IMDL-BenCo library</span></span>
<span class="line">pip uninstall imdlbenco</span>
<span class="line"></span>
<span class="line"><span class="token comment"># Clone your forked IMDL-BenCo repository from GitHub</span></span>
<span class="line"><span class="token function">git</span> clone https://github.com/your_name/IMDL-BenCo.git</span>
<span class="line"></span>
<span class="line"><span class="token comment"># Enter the project directory</span></span>
<span class="line"><span class="token builtin class-name">cd</span> IMDL-BenCo</span>
<span class="line"></span>
<span class="line"><span class="token comment"># Perform a local development installation using \`pip install -e .\`</span></span>
<span class="line">pip <span class="token function">install</span> <span class="token parameter variable">-e</span> <span class="token builtin class-name">.</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Verify the installation</span></span>
<span class="line">pip show imdlbenco</span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>If the installation is successful, after executing <code>pip list</code>, you should see something like this:</p><div class="language-text line-numbers-mode" data-highlighter="prismjs" data-ext="text"><pre><code><span class="line">Package                 Version            Editable project location</span>
<span class="line">----------------------- ------------------ ------------------------------------------------------</span>
<span class="line">...</span>
<span class="line">IMDLBenCo               0.1.10             /mnt/data0/xiaochen/workspace/IMDLBenCo_pure/IMDLBenCo</span>
<span class="line">...</span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>The presence of a corresponding path in the <code>Editable project location</code> column indicates that any modifications to the Python scripts in this path will take effect directly in the Python environment without the need for reinstallation. This is very convenient for debugging.</p>`,9))])}const g=r(u,[["render",h]]),f=JSON.parse('{"path":"/guide/quickstart/install.html","title":"Installation","lang":"en-US","frontmatter":{"description":"Installation Please note that, unlike most code provided in papers, the Benco repository is not intended to be used via methods like git clone, as the code involves numerous com...","head":[["link",{"rel":"alternate","hreflang":"zh-cn","href":"https://scu-zjz.github.io/IMDLBenCo-doc/IMDLBenCo-doc/zh/guide/quickstart/install.html"}],["meta",{"property":"og:url","content":"https://scu-zjz.github.io/IMDLBenCo-doc/IMDLBenCo-doc/guide/quickstart/install.html"}],["meta",{"property":"og:site_name","content":"IMDLBenCo Documentation"}],["meta",{"property":"og:title","content":"Installation"}],["meta",{"property":"og:description","content":"Installation Please note that, unlike most code provided in papers, the Benco repository is not intended to be used via methods like git clone, as the code involves numerous com..."}],["meta",{"property":"og:type","content":"article"}],["meta",{"property":"og:locale","content":"en-US"}],["meta",{"property":"og:locale:alternate","content":"zh-CN"}],["meta",{"property":"og:updated_time","content":"2025-03-30T13:14:09.000Z"}],["meta",{"property":"article:modified_time","content":"2025-03-30T13:14:09.000Z"}],["script",{"type":"application/ld+json"},"{\\"@context\\":\\"https://schema.org\\",\\"@type\\":\\"Article\\",\\"headline\\":\\"Installation\\",\\"image\\":[\\"\\"],\\"dateModified\\":\\"2025-03-30T13:14:09.000Z\\",\\"author\\":[]}"]]},"headers":[{"level":2,"title":"For Regular Users","slug":"for-regular-users","link":"#for-regular-users","children":[]},{"level":2,"title":"For Developers Contributing to the Official Repository","slug":"for-developers-contributing-to-the-official-repository","link":"#for-developers-contributing-to-the-official-repository","children":[]}],"git":{"updatedTime":1743340449000,"contributors":[{"name":"Ma Xiaochen (马晓晨)","username":"","email":"mxch1122@126.com","commits":7}],"changelog":[{"hash":"e7e94b639b491b3bf1b5f0d88e23bc7b0135d8f1","time":1743340449000,"email":"mxch1122@126.com","author":"Ma Xiaochen (马晓晨)","message":"[update] add english version for demo. revise readme. remove redundant line in install"},{"hash":"5398ea0cf41847f4d5f5fd0ebbe3bcbb5dec7114","time":1743339971000,"email":"mxch1122@126.com","author":"Ma Xiaochen (马晓晨)","message":"[update] update version and modify benco -v"},{"hash":"6db2d2f20cb8ff2505eff176cbe1d5c02670f23e","time":1737960496000,"email":"mxch1122@126.com","author":"Ma Xiaochen (马晓晨)","message":"[update] add benco --version check"},{"hash":"dd8ba6220fff989633869274cd8b32fe443434e7","time":1729058253000,"email":"mxch1122@126.com","author":"Ma Xiaochen (马晓晨)","message":"[update] revise phrasing in install.md"},{"hash":"8c26dc30c467afbc8ec70ff807ceaca6c1a0e595","time":1725040789000,"email":"mxch1122@126.com","author":"Ma Xiaochen (马晓晨)","message":"[update] add document for model_zoo"},{"hash":"6de34ba84e800741a8134cad3c0a4c9925e096ef","time":1724932687000,"email":"mxch1122@126.com","author":"Ma Xiaochen (马晓晨)","message":"[update] add guide and dataprepare process"},{"hash":"a8f6cb56d3483ba336ed64a05ab07ab637eb4db7","time":1719873732000,"email":"mxch1122@126.com","author":"Ma Xiaochen (马晓晨)","message":"[update] add install for developer"}]},"filePathRelative":"guide/quickstart/install.md","autoDesc":true}');export{g as comp,f as data};
