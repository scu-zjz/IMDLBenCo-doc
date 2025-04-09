import{_ as a}from"./demo--WtJ5ARF.js";import{_ as e,c as t,a as i,e as o,r as p,o as l}from"./app-DCerHgAi.js";const c={};function r(d,n){const s=p("CommentService");return l(),t("div",null,[n[0]||(n[0]=i('<h1 id="dataset-preparation" tabindex="-1"><a class="header-anchor" href="#dataset-preparation"><span>Dataset Preparation</span></a></h1><h2 id="important" tabindex="-1"><a class="header-anchor" href="#important"><span>Important</span></a></h2><p>The functionality and interfaces of the dataset section will be managed by the benco CLI in subsequent versions.</p><p>For now, it is temporarily necessary to manually manage the corresponding <code>json</code> or dataset paths in each working path to complete the deployment.</p><h2 id="tampering-detection-task-dataset-introduction" tabindex="-1"><a class="header-anchor" href="#tampering-detection-task-dataset-introduction"><span>Tampering Detection Task Dataset Introduction</span></a></h2><ul><li>Currently, tampering detection generally includes two types of tasks: <ul><li><strong>Organized in Detection form</strong>, perform image-level binary classification on a whole image to determine whether the image is tampered with.</li><li><strong>Organized in Segmentation form</strong>, generate a pixel-level binary classification mask for an image to segment the tampered area.</li></ul></li><li>Therefore, generally speaking, a record in a tampering detection dataset includes the following content: <ul><li>A tampered image, image</li><li>A corresponding binary mask of the tampered area</li><li>A 0, 1 label representing whether the image has been tampered with.</li></ul></li><li>Below are two typical pairs of tampered images and their corresponding masks: <ul><li><img src="'+a+`" alt=""></li></ul></li><li>Many papers only use &quot;datasets that only contain tampered images&quot;. Recently, some papers have tried to introduce real images for training. Although this can reduce the false positive rate, it will cause a slight decrease in overall metrics (the model will tend not to predict, missing some positive points).</li></ul><h2 id="dataset-format-and-features" tabindex="-1"><a class="header-anchor" href="#dataset-format-and-features"><span>Dataset Format and Features</span></a></h2><ul><li>IMDL-BenCo internally implements three different dataset formats, corresponding to different dataset organization methods. Various tampering datasets can be organized into these formats for the framework to read.</li><li>The preset dataset formats of IMDL-BenCo include two basic <code>JsonDataset</code> and <code>ManiDataset</code>, used for reading individual datasets. There is also a <code>BalanceDataset</code>, which manages multiple datasets according to a special sampling strategy. Organizing the dataset in any of these three ways allows it to be read by IMDL-BenCo. Their specific introductions are as follows: <ul><li><code>ManiDataset</code>, automatically reads all images in two folders (named <code>./Tp</code> and <code>./Gt</code>) under a path, serving as the image to be tested and the corresponding mask. Suitable for lightweight development and occasions where <strong>real images do not need to be introduced</strong>.</li><li><code>JsonDataset</code>, indexes the paths of required data through a Json file, suitable for occasions where <strong>real images need to be introduced</strong>.</li><li><code>BalancedDataset</code>, this dataset manages a dictionary that stores multiple <code>ManiDataset</code> or <code>JsonDataset</code> objects, and randomly samples n images from all the sub-datasets it contains in each Epoch (default only samples 1800 images). Therefore, the actual number of images participating in training in one Epoch is <strong>the number of datasets × n</strong>, but when the dataset is large enough, the richness of images over multiple Epochs can still be high. Moreover, it avoids the model trained after being too &quot;overfitted&quot; to large datasets. <code>BalancedDataset</code> is mainly designed for the protocols of <a href="https://openaccess.thecvf.com/content/WACV2021/html/Kwon_CAT-Net_Compression_Artifact_Tracing_Network_for_Detection_and_Localization_of_WACV_2021_paper.html" target="_blank" rel="noopener noreferrer">CAT-Net</a> and <a href="https://openaccess.thecvf.com/content/CVPR2023/html/Guillaro_TruFor_Leveraging_All-Round_Clues_for_Trustworthy_Image_Forgery_Detection_and_CVPR_2023_paper.html" target="_blank" rel="noopener noreferrer">TruFor</a>. If you are not reproducing the protocol for this agreement, you do not need to pay attention.</li></ul></li></ul><p>The above datasets can be used for direct training or testing. In addition, to improve efficiency in testing, multiple different datasets can be tested in sequence in one round of scripts, so an additional Json format is defined for inputting a large number of datasets, with an example at the end of this section.</p><h2 id="specific-definition-format" tabindex="-1"><a class="header-anchor" href="#specific-definition-format"><span>Specific Definition Format</span></a></h2><ol><li><p><code>ManiDataset</code>, <strong>pass in a folder path</strong>, the folder contains two sub-folders <code>Tp</code> and <code>Gt</code>, benco automatically reads images from <code>Tp</code>, reads corresponding masks from <code>Gt</code>, and automatically pairs all image files in the two folders according to <strong>dictionary order</strong> to obtain a complete dataset. You can refer to the <a href="https://github.com/SunnyHaze/IML-ViT/tree/main/images/sample_iml_dataset" target="_blank" rel="noopener noreferrer">IML-ViT sample folder</a>.</p></li><li><p><code>JsonDataset</code>, <strong>pass in a JSON file path</strong>, organize images and corresponding masks with the following JSON format:</p><div class="language-text line-numbers-mode" data-highlighter="prismjs" data-ext="text"><pre><code><span class="line">[</span>
<span class="line">    [</span>
<span class="line">      &quot;/Dataset/CASIAv2/Tp/Tp_D_NRN_S_N_arc00013_sec00045_11700.jpg&quot;,</span>
<span class="line">      &quot;/Dataset/CASIAv2/Gt/Tp_D_NRN_S_N_arc00013_sec00045_11700_gt.png&quot;</span>
<span class="line">    ],</span>
<span class="line">    ......</span>
<span class="line">    [</span>
<span class="line">      &quot;/Dataset/CASIAv2/Au/Au_nat_30198.jpg&quot;,</span>
<span class="line">      &quot;Negative&quot;</span>
<span class="line">    ],</span>
<span class="line">    ......</span>
<span class="line">]</span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>Where &quot;Negative&quot; indicates a completely black mask, i.e., a completely real image, so there is no need to input the path.</p></li><li><p><code>BalancedDataset</code>, pass in a JSON file path, used to organize and generate multiple sub-datasets, and sample from these sub-datasets when used. Specifically for organizing the protocols used in <a href="https://openaccess.thecvf.com/content/WACV2021/html/Kwon_CAT-Net_Compression_Artifact_Tracing_Network_for_Detection_and_Localization_of_WACV_2021_paper.html" target="_blank" rel="noopener noreferrer">CAT-Net</a> and <a href="https://openaccess.thecvf.com/content/CVPR2023/html/Guillaro_TruFor_Leveraging_All-Round_Clues_for_Trustworthy_Image_Forgery_Detection_and_CVPR_2023_paper.html" target="_blank" rel="noopener noreferrer">TruFor</a>.</p><ol><li>Specific protocol definition: Protocol-CAT uses 9 large datasets for training, but only randomly samples 1800 images from each dataset to form a 16200-image dataset for training in each Epoch.</li><li>Json organization form:</li></ol><div class="language-JSON line-numbers-mode" data-highlighter="prismjs" data-ext="JSON"><pre><code><span class="line">[</span>
<span class="line">   [</span>
<span class="line">       &quot;ManiDataset&quot;,</span>
<span class="line">       &quot;/mnt/data0/public_datasets/IML/CASIA2.0&quot;</span>
<span class="line">   ],</span>
<span class="line">   [</span>
<span class="line">       &quot;JsonDataset&quot;,</span>
<span class="line">       &quot;/mnt/data0/public_datasets/IML/FantasticReality_v1/FantasticReality.json&quot;</span>
<span class="line">   ],</span>
<span class="line">   [</span>
<span class="line">       &quot;ManiDataset&quot;,</span>
<span class="line">       &quot;/mnt/data0/public_datasets/IML/IMD_20_1024&quot;</span>
<span class="line">   ],</span>
<span class="line">   [</span>
<span class="line">        &quot;JsonDataset&quot;,</span>
<span class="line">        &quot;/mnt/data0/public_datasets/IML/compRAISE/compRAISE_1024_list.json&quot;</span>
<span class="line">  ],</span>
<span class="line">   [</span>
<span class="line">       &quot;JsonDataset&quot;,</span>
<span class="line">       &quot;/mnt/data0/public_datasets/IML/tampCOCO/sp_COCO_list.json&quot;</span>
<span class="line">   ],</span>
<span class="line">   [</span>
<span class="line">       &quot;JsonDataset&quot;,</span>
<span class="line">       &quot;/mnt/data0/public_datasets/IML/tampCOCO/cm_COCO_list.json&quot;</span>
<span class="line">   ],</span>
<span class="line">   [</span>
<span class="line">       &quot;JsonDataset&quot;,</span>
<span class="line">       &quot;/mnt/data0/public_datasets/IML/tampCOCO/bcm_COCO_list.json&quot;</span>
<span class="line">   ],</span>
<span class="line">   [</span>
<span class="line">       &quot;JsonDataset&quot;,</span>
<span class="line">       &quot;/mnt/data0/public_datasets/IML/tampCOCO/bcmc_COCO_list.json&quot;</span>
<span class="line">   ]</span>
<span class="line">]</span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>A two-dimensional array, each row represents a dataset, the first column represents the string of the dataset Class type used (read the corresponding dataset according to the organization method of <code>ManiDataset</code> or <code>JsonDataset</code>), and the second column is the path of the dataset that needs to be read for this type.</p></li></ol><p>Organize the datasets needed according to the requirements, and then you can start considering reproducing the model or implementing your own model.</p><p>In addition to the format to be noted, to improve the speed of training and testing, it is also necessary to perform necessary preprocessing on the images.</p><h2 id="preprocessing-for-high-resolution-images" tabindex="-1"><a class="header-anchor" href="#preprocessing-for-high-resolution-images"><span>Preprocessing for High-Resolution Images</span></a></h2><p>Some datasets have very high resolutions, such as the NIST16 and compRAISE datasets in the CAT-Protocol, which contain 4000x4000 images. These datasets, if directly read during training, will bring a very high I/O burden. Especially when used as training datasets.</p><p>So we particularly recommend resizing the images to a smaller size in advance when using these datasets, such as reducing to a long side equal to 1024 while maintaining the aspect ratio. Otherwise, the training speed may be greatly slowed down, please refer to <a href="https://github.com/scu-zjz/IMDLBenCo/issues/40" target="_blank" rel="noopener noreferrer">IMDL-BenCo issue #40</a>.</p><p>We provide a Resize code based on a thread pool here, which can efficiently convert all images in a path to the desired resolution through multi-threading:</p><div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre><code><span class="line"><span class="token keyword">import</span> os</span>
<span class="line"><span class="token keyword">from</span> PIL <span class="token keyword">import</span> Image</span>
<span class="line"><span class="token keyword">from</span> concurrent<span class="token punctuation">.</span>futures <span class="token keyword">import</span> ThreadPoolExecutor</span>
<span class="line"></span>
<span class="line"><span class="token keyword">def</span> <span class="token function">process_image</span><span class="token punctuation">(</span>filename<span class="token punctuation">,</span> directory<span class="token punctuation">,</span> output_directory<span class="token punctuation">,</span> target_size<span class="token punctuation">)</span><span class="token punctuation">:</span></span>
<span class="line">    <span class="token keyword">try</span><span class="token punctuation">:</span></span>
<span class="line">        <span class="token keyword">with</span> Image<span class="token punctuation">.</span><span class="token builtin">open</span><span class="token punctuation">(</span>os<span class="token punctuation">.</span>path<span class="token punctuation">.</span>join<span class="token punctuation">(</span>directory<span class="token punctuation">,</span> filename<span class="token punctuation">)</span><span class="token punctuation">)</span> <span class="token keyword">as</span> img<span class="token punctuation">:</span></span>
<span class="line">            width<span class="token punctuation">,</span> height <span class="token operator">=</span> img<span class="token punctuation">.</span>size</span>
<span class="line">            <span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f&#39;Processing Image: </span><span class="token interpolation"><span class="token punctuation">{</span>filename<span class="token punctuation">}</span></span><span class="token string"> | Resolution: </span><span class="token interpolation"><span class="token punctuation">{</span>width<span class="token punctuation">}</span></span><span class="token string">x</span><span class="token interpolation"><span class="token punctuation">{</span>height<span class="token punctuation">}</span></span><span class="token string">&#39;</span></span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line">            <span class="token comment"># Determine the scaling ratio with the long side as 1024</span></span>
<span class="line">            <span class="token keyword">if</span> <span class="token builtin">max</span><span class="token punctuation">(</span>width<span class="token punctuation">,</span> height<span class="token punctuation">)</span> <span class="token operator">&gt;</span> target_size<span class="token punctuation">:</span></span>
<span class="line">                <span class="token keyword">if</span> width <span class="token operator">&gt;</span> height<span class="token punctuation">:</span></span>
<span class="line">                    new_width <span class="token operator">=</span> target_size</span>
<span class="line">                    new_height <span class="token operator">=</span> <span class="token builtin">int</span><span class="token punctuation">(</span><span class="token punctuation">(</span>target_size <span class="token operator">/</span> width<span class="token punctuation">)</span> <span class="token operator">*</span> height<span class="token punctuation">)</span></span>
<span class="line">                <span class="token keyword">else</span><span class="token punctuation">:</span></span>
<span class="line">                    new_height <span class="token operator">=</span> target_size</span>
<span class="line">                    new_width <span class="token operator">=</span> <span class="token builtin">int</span><span class="token punctuation">(</span><span class="token punctuation">(</span>target_size <span class="token operator">/</span> height<span class="token punctuation">)</span> <span class="token operator">*</span> width<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line">                <span class="token comment"># Resize the image</span></span>
<span class="line">                img_resized <span class="token operator">=</span> img<span class="token punctuation">.</span>resize<span class="token punctuation">(</span><span class="token punctuation">(</span>new_width<span class="token punctuation">,</span> new_height<span class="token punctuation">)</span><span class="token punctuation">,</span> Image<span class="token punctuation">.</span>ANTIALIAS<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line">                <span class="token comment"># Save the image to the specified folder</span></span>
<span class="line">                output_path <span class="token operator">=</span> os<span class="token punctuation">.</span>path<span class="token punctuation">.</span>join<span class="token punctuation">(</span>output_directory<span class="token punctuation">,</span> filename<span class="token punctuation">)</span></span>
<span class="line">                img_resized<span class="token punctuation">.</span>save<span class="token punctuation">(</span>output_path<span class="token punctuation">)</span></span>
<span class="line">                <span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f&#39;Resized and saved </span><span class="token interpolation"><span class="token punctuation">{</span>filename<span class="token punctuation">}</span></span><span class="token string"> to </span><span class="token interpolation"><span class="token punctuation">{</span>output_directory<span class="token punctuation">}</span></span><span class="token string"> with resolution </span><span class="token interpolation"><span class="token punctuation">{</span>new_width<span class="token punctuation">}</span></span><span class="token string">x</span><span class="token interpolation"><span class="token punctuation">{</span>new_height<span class="token punctuation">}</span></span><span class="token string">&#39;</span></span><span class="token punctuation">)</span></span>
<span class="line">            <span class="token keyword">else</span><span class="token punctuation">:</span></span>
<span class="line">                <span class="token comment"># If the image does not need to be adjusted, directly copy it to the target folder</span></span>
<span class="line">                img<span class="token punctuation">.</span>save<span class="token punctuation">(</span>os<span class="token punctuation">.</span>path<span class="token punctuation">.</span>join<span class="token punctuation">(</span>output_directory<span class="token punctuation">,</span> filename<span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">                <span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f&#39;Image </span><span class="token interpolation"><span class="token punctuation">{</span>filename<span class="token punctuation">}</span></span><span class="token string"> already meets the target size and was saved without resizing.&#39;</span></span><span class="token punctuation">)</span></span>
<span class="line">            <span class="token keyword">return</span> <span class="token number">1</span>  <span class="token comment"># Return the count of successful processing</span></span>
<span class="line">    <span class="token keyword">except</span> Exception <span class="token keyword">as</span> e<span class="token punctuation">:</span></span>
<span class="line">        <span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f&quot;Cannot process </span><span class="token interpolation"><span class="token punctuation">{</span>filename<span class="token punctuation">}</span></span><span class="token string">: </span><span class="token interpolation"><span class="token punctuation">{</span>e<span class="token punctuation">}</span></span><span class="token string">&quot;</span></span><span class="token punctuation">)</span></span>
<span class="line">        <span class="token keyword">return</span> <span class="token number">0</span>  <span class="token comment"># Return the count of failed processing</span></span>
<span class="line"></span>
<span class="line"><span class="token keyword">def</span> <span class="token function">get_image_resolutions_and_resize</span><span class="token punctuation">(</span>directory<span class="token operator">=</span><span class="token string">&#39;.&#39;</span><span class="token punctuation">,</span> output_directory<span class="token operator">=</span><span class="token string">&#39;resized_images&#39;</span><span class="token punctuation">,</span> target_size<span class="token operator">=</span><span class="token number">1024</span><span class="token punctuation">)</span><span class="token punctuation">:</span></span>
<span class="line">    <span class="token comment"># Create the output folder, create if it does not exist</span></span>
<span class="line">    <span class="token keyword">if</span> <span class="token keyword">not</span> os<span class="token punctuation">.</span>path<span class="token punctuation">.</span>exists<span class="token punctuation">(</span>output_directory<span class="token punctuation">)</span><span class="token punctuation">:</span></span>
<span class="line">        os<span class="token punctuation">.</span>makedirs<span class="token punctuation">(</span>output_directory<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line">    <span class="token comment"># Get all image files</span></span>
<span class="line">    image_files <span class="token operator">=</span> <span class="token punctuation">[</span>f <span class="token keyword">for</span> f <span class="token keyword">in</span> os<span class="token punctuation">.</span>listdir<span class="token punctuation">(</span>directory<span class="token punctuation">)</span> <span class="token keyword">if</span> f<span class="token punctuation">.</span>lower<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">.</span>endswith<span class="token punctuation">(</span><span class="token punctuation">(</span><span class="token string">&#39;png&#39;</span><span class="token punctuation">,</span> <span class="token string">&#39;jpg&#39;</span><span class="token punctuation">,</span> <span class="token string">&#39;jpeg&#39;</span><span class="token punctuation">,</span> <span class="token string">&#39;bmp&#39;</span><span class="token punctuation">,</span> <span class="token string">&#39;gif&#39;</span><span class="token punctuation">,</span> <span class="token string">&#39;tiff&#39;</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">]</span></span>
<span class="line">    </span>
<span class="line">    <span class="token comment"># Use a thread pool to process images</span></span>
<span class="line">    total_processed <span class="token operator">=</span> <span class="token number">0</span></span>
<span class="line">    <span class="token keyword">with</span> ThreadPoolExecutor<span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token keyword">as</span> executor<span class="token punctuation">:</span></span>
<span class="line">        futures <span class="token operator">=</span> <span class="token punctuation">[</span>executor<span class="token punctuation">.</span>submit<span class="token punctuation">(</span>process_image<span class="token punctuation">,</span> filename<span class="token punctuation">,</span> directory<span class="token punctuation">,</span> output_directory<span class="token punctuation">,</span> target_size<span class="token punctuation">)</span> <span class="token keyword">for</span> filename <span class="token keyword">in</span> image_files<span class="token punctuation">]</span></span>
<span class="line">        </span>
<span class="line">        <span class="token comment"># Wait for all threads to complete and accumulate the number of processed</span></span>
<span class="line">        <span class="token keyword">for</span> future <span class="token keyword">in</span> futures<span class="token punctuation">:</span></span>
<span class="line">            total_processed <span class="token operator">+=</span> future<span class="token punctuation">.</span>result<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line">    <span class="token comment"># Output the total number of images</span></span>
<span class="line">    <span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f&quot;\\nTotal number of images processed: </span><span class="token interpolation"><span class="token punctuation">{</span>total_processed<span class="token punctuation">}</span></span><span class="token string">&quot;</span></span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Execute the function</span></span>
<span class="line">get_image_resolutions_and_resize<span class="token punctuation">(</span></span>
<span class="line">    directory<span class="token operator">=</span><span class="token string">&quot;./compRAISE&quot;</span><span class="token punctuation">,</span></span>
<span class="line">    output_directory<span class="token operator">=</span><span class="token string">&quot;./compRAISE1024&quot;</span><span class="token punctuation">,</span></span>
<span class="line">    target_size<span class="token operator">=</span><span class="token number">1024</span></span>
<span class="line"><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h2 id="test-dataset-json" tabindex="-1"><a class="header-anchor" href="#test-dataset-json"><span>Test Dataset JSON</span></a></h2><p>Specifically, for testing, since batch testing needs to be completed on multiple datasets, a <code>test_dataset.json</code> is defined to accomplish this function. Because it is the testing phase, only paths representing <code>ManiDataset</code> or <code>JsonDataset</code> can be passed as test sets; different from <code>BalancedDataset</code>, which can only be used for training.</p><p>The Key is the field name used for Tensorboard, log output, and other Visualize features, and the Value is the specific path of the above datasets.</p><p>An example of <code>test_datasets.json</code>, directly pass the path of this json to the training script as the test set (introduced later):</p><div class="language-JSON line-numbers-mode" data-highlighter="prismjs" data-ext="JSON"><pre><code><span class="line">{</span>
<span class="line">    &quot;Columbia&quot;: &quot;/mnt/data0/public_datasets/IML/Columbia.json&quot;,</span>
<span class="line">    &quot;NIST16_1024&quot;: &quot;/mnt/data0/public_datasets/IML/NIST16_1024&quot;,</span>
<span class="line">    &quot;NIST16_cleaned&quot;: &quot;/mnt/data0/public_datasets/IML/NIST16_1024_cleaning&quot;,</span>
<span class="line">    &quot;coverage&quot;: &quot;/mnt/data0/public_datasets/IML/coverage.json&quot;,</span>
<span class="line">    &quot;CASIAv1&quot;: &quot;/mnt/data0/public_datasets/IML/CASIA1.0&quot;,</span>
<span class="line">    &quot;IMD20_1024&quot;: &quot;/mnt/data0/public_datasets/IML/IMD_20_1024&quot;</span>
<span class="line">}</span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>`,23)),o(s)])}const h=e(c,[["render",r]]),g=JSON.parse('{"path":"/guide/quickstart/0_dataprepare.html","title":"Dataset Preparation","lang":"en-US","frontmatter":{"description":"Dataset Preparation Important The functionality and interfaces of the dataset section will be managed by the benco CLI in subsequent versions. For now, it is temporarily necessa...","head":[["link",{"rel":"alternate","hreflang":"zh-cn","href":"https://scu-zjz.github.io/IMDLBenCo-doc/IMDLBenCo-doc/zh/guide/quickstart/0_dataprepare.html"}],["meta",{"property":"og:url","content":"https://scu-zjz.github.io/IMDLBenCo-doc/IMDLBenCo-doc/guide/quickstart/0_dataprepare.html"}],["meta",{"property":"og:site_name","content":"IMDLBenCo Documentation"}],["meta",{"property":"og:title","content":"Dataset Preparation"}],["meta",{"property":"og:description","content":"Dataset Preparation Important The functionality and interfaces of the dataset section will be managed by the benco CLI in subsequent versions. For now, it is temporarily necessa..."}],["meta",{"property":"og:type","content":"article"}],["meta",{"property":"og:image","content":"https://scu-zjz.github.io/IMDLBenCo-doc/IMDLBenCo-doc/images/assets/demo.png"}],["meta",{"property":"og:locale","content":"en-US"}],["meta",{"property":"og:locale:alternate","content":"zh-CN"}],["meta",{"property":"og:updated_time","content":"2025-04-03T10:12:18.000Z"}],["meta",{"property":"article:modified_time","content":"2025-04-03T10:12:18.000Z"}],["script",{"type":"application/ld+json"},"{\\"@context\\":\\"https://schema.org\\",\\"@type\\":\\"Article\\",\\"headline\\":\\"Dataset Preparation\\",\\"image\\":[\\"https://scu-zjz.github.io/IMDLBenCo-doc/IMDLBenCo-doc/images/assets/demo.png\\"],\\"dateModified\\":\\"2025-04-03T10:12:18.000Z\\",\\"author\\":[]}"]]},"headers":[{"level":2,"title":"Important","slug":"important","link":"#important","children":[]},{"level":2,"title":"Tampering Detection Task Dataset Introduction","slug":"tampering-detection-task-dataset-introduction","link":"#tampering-detection-task-dataset-introduction","children":[]},{"level":2,"title":"Dataset Format and Features","slug":"dataset-format-and-features","link":"#dataset-format-and-features","children":[]},{"level":2,"title":"Specific Definition Format","slug":"specific-definition-format","link":"#specific-definition-format","children":[]},{"level":2,"title":"Preprocessing for High-Resolution Images","slug":"preprocessing-for-high-resolution-images","link":"#preprocessing-for-high-resolution-images","children":[]},{"level":2,"title":"Test Dataset JSON","slug":"test-dataset-json","link":"#test-dataset-json","children":[]}],"git":{"updatedTime":1743675138000,"contributors":[{"name":"Ma Xiaochen (马晓晨)","username":"","email":"mxch1122@126.com","commits":6}],"changelog":[{"hash":"3adb8aa2f1fd3408eb9837aa011bf7048ae6230b","time":1743675138000,"email":"mxch1122@126.com","author":"Ma Xiaochen (马晓晨)","message":"[update] add comment plugin to all pages."},{"hash":"2e30180fb8a110e32a3aa5fe308d2e0a030c47ec","time":1743364035000,"email":"mxch1122@126.com","author":"Ma Xiaochen (马晓晨)","message":"Revise sturcture, add more infor about data &amp; models."},{"hash":"9e1ccd549ba841731ff39d78ab11cd08d1c26564","time":1743322393000,"email":"mxch1122@126.com","author":"Ma Xiaochen (马晓晨)","message":"[update] revise description reltated to BalancedDatasets."},{"hash":"ccd6ad1772c39ae5a7d55b205e729e92aa5d4ac5","time":1727242612000,"email":"mxch1122@126.com","author":"Ma Xiaochen (马晓晨)","message":"[update] add compRAISE_1024.json to guidance"},{"hash":"2c864183d9a9f6b2d1b517189d46c6a27a3ba2a2","time":1727241998000,"email":"mxch1122@126.com","author":"Ma Xiaochen (马晓晨)","message":"[update] add dataset preprocessing guide"},{"hash":"8c26dc30c467afbc8ec70ff807ceaca6c1a0e595","time":1725040789000,"email":"mxch1122@126.com","author":"Ma Xiaochen (马晓晨)","message":"[update] add document for model_zoo"}]},"filePathRelative":"guide/quickstart/0_dataprepare.md","autoDesc":true}');export{h as comp,g as data};
