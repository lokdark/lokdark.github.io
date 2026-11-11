# Luke's Blog

## 目录结构

| 路径 | 说明 |
| --- | --- |
| `hugo.toml` | 站点配置：标题、作者、菜单、社交链接、`baseURL`、页脚等 |
| `content/` | 文章与页面（Markdown） |
| `content/posts/` | 博客文章 |
| `content/about.md`、`content/about/me.md` | About / About Me 页面（About Me 不在导航里，从 About 页进入） |
| `archetypes/default.md`、`archetypes/posts.md` | `hugo new` 新建文件时套用的模板（`content/posts/` 下用 `posts.md`，自带 `author: Luke`） |
| `layouts/partials/extra-head.html` | 站点级覆写：把默认配色固定为暗色（见下方「配色」） |
| `layouts/index.html` | 站点级覆写：首页 = 头像 + 大标题 + 打字机副标题 + 分类卡片 + 最新文章 |
| `content/categories/<key>/_index.md` | 分类页：`title` / `description` / `weight`，首页分类卡片由它生成（见下方「分类」） |
| `static/css/Luke.css` | 站点级样式：首页卡片、社交图标行、头像修饰（通过 `hugo.toml` 的 `customCSS` 挂载） |
| `static/img/avatar.jpg` | 首页头像（`[params.portrait]`），480×480 正方形；换图直接替换此文件或改 `path` |
| `layouts/partials/subtitle.html` | 站点级覆写：首页副标题的打字机动画（见下方「打字机效果」） |
| `themes/hello-friend-ng/` | 主题源码（MIT License），原样保留，可自行修改或整体替换 |
| `.github/workflows/deploy.yml` | push 到 `main` 后自动构建并发布到 GitHub Pages |
| `.tools/` | 本地预览用的 Hugo 可执行文件（已在 `.gitignore` 中忽略，不会提交） |

## 本地预览

主题使用 SCSS，必须用 **Hugo extended** 版本（仓库内自带的即为 extended 版）。

```powershell
# 使用仓库内自带的 Hugo
.\.tools\hugo.exe server -D

# 或使用系统已安装的 hugo
hugo server -D
```

然后打开 <http://localhost:1313/> 。`-D` 表示同时渲染 `draft: true` 的草稿文章。

只生成静态文件（输出到 `public/`，该目录不提交）：

```powershell
.\.tools\hugo.exe --gc --minify
```

## 新建文章

```powershell
.\.tools\hugo.exe new posts/my-first-post.md
```

文章会带上 `draft: true`，写完把 `draft` 改成 `false`（或本地预览时加 `-D`）。

## 部署到 GitHub Pages

1. 提交并 push 到 `main` 分支。
2. 仓库 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**（只需设置一次）。
3. `Deploy Hugo site to GitHub Pages` 这个 workflow 会自动构建并发布，站点地址见 workflow 的 `deploy` 步骤输出。

`hugo.toml` 中的 `baseURL` 只影响本地构建；CI 中会用 `actions/configure-pages` 输出的真实站点地址覆盖它。

## 配色（暗色 / 浅色）

主题原生是「跟随操作系统」的：`prefers-color-scheme` 是浅色时页面就是白底。

本站通过 `layouts/partials/extra-head.html`（主题预留的扩展点）在页面绘制前给 `<html>` 打上
`data-theme="dark"`，所以**默认固定为官方演示那种黑灰暗色**，与访客系统设置无关，也不会出现白屏闪烁。

- 想默认改成浅色：把 `layouts/partials/extra-head.html` 里的 `"dark"` 改成 `"light"`。
- 想恢复跟随系统：删掉 `layouts/partials/extra-head.html` 即可。
- 想让访客自己切换：`hugo.toml` 里把 `enableThemeToggle` 设为 `true`（当前为 `false`，页头右侧没有切换按钮），
  按钮出现后选择结果会记在浏览器 localStorage 里。

## 首页副标题打字机效果

`layouts/partials/subtitle.html` 覆写了主题的同名 partial（只作用于首页），把
`hugo.toml` 中 `homeSubtitle` 的文案按字符逐个「打」出来，末尾带一个粉色闪烁光标
（配色与页头 logo 光标一致）。**默认已开启循环**：打完停顿 2 秒 → 逐字回删 → 再打，
如此往复。

首页大标题（`<h1>`）由 `layouts/index.html` 覆写提供，取值顺序是
`params.homeTitle` → 站点 `title`。

可调参数就在 `layouts/partials/subtitle.html` 的 `<span class="home-subtitle__text">` 上：

| 属性 | 当前值 | 说明 |
| --- | --- | --- |
| `data-typing-delay` | `350` | 开始打字前的等待毫秒数（循环时也是每轮开始前的等待） |
| `data-typing-speed` | `65` | 每个字符的毫秒数 |
| `data-typing-loop` | `true` | `true` = 打完删除再重打；`false` = 只打一次 |
| `data-typing-cursor` | `true` | `false` 表示不显示光标 |

细节：禁用 JavaScript 时按原样直接显示整句文案；系统开启「减少动态效果」
（`prefers-reduced-motion`）时跳过动画。删除该文件即可恢复主题默认的静态副标题。

目前循环的是同一句文案。如果想做多句轮播（例如依次打 "Hello" / "Welcome" / "Enjoy"），
需要给这个 partial 加一段句子列表的逻辑，需要的话我可以改。
## 一个容易踩的坑：canonifyURLs

`hugo.toml` 里 `canonifyURLs` 必须保持 `false`。主题的 exampleSite 默认是 `true`，
它会把 CSS/JS 等资源链接改写成 `https://你的域名/...` 的绝对地址，结果是
**本地 `hugo server` 预览时样式完全不生效**（页面白底、无排版）。设为 `false` 后
资源用站点根相对路径（如 `/main.min.xxx.css`），本地和线上都正常。

## 后续可自行修改的地方

- `hugo.toml`：`baseURL`、`title`、`homeSubtitle`、`params.author.name`、`[[params.social]]`、
  `params.logo.logoText`、`params.footer.*`。
- `content/about.md`、`content/about/me.md`：个人介绍（About Me 页面仍在，只是不再放在导航里，可从 About 页进入）。
- 首页头像：`hugo.toml` 里 `[params.portrait]`（`path` / `alt` / `maxWidth`），删掉整块即不显示。图片要求正方形，建议 400px 以上。
- 首页卡片样式：`static/css/Luke.css`，`.home-cards` 的 `max-width`、`gap` 可调。
- 社交图标：`hugo.toml` 的 `[[params.social]]`，`name` 取值见 `themes/hello-friend-ng/docs/svgs.md`。
- `content/posts/hello-world.md`：Hugo 官方自带的 4 篇示例文章已删除，这里留了一篇空文章当样板；
  写新文章时直接复制它的 front matter 即可。
- `[menu]`：页头导航是 About / Categories / Posts（顺序按主题的 `identifier` 字母序）。About 早先是悬停下拉菜单（所以点它没反应），现已改成直接跳 `/about/` 的普通链接。
- `[params] enableGlobalLanguageMenu`：语言切换菜单（页头右侧的 `en`）已关闭，站点只有一种语言时用不到。
- 绑定自定义域名：把域名写进 `static/CNAME`，并把 `baseURL` 改为该域名。
- 评论区：`hugo.toml` 里已预留 `[params.utterances]` 配置（基于 GitHub Issues），需要时取消注释并填上 `repository`。

## 站点图标（favicon）

图标放在 `static/` 下，配色取自站点本身：底色 `#1b1c1d`（页头暗色）+ 主色 `#fe5186`（主题强调色，
和页头光标、`>` 提示符一致），图形是 Feather Icons 的 `chevron-right`（MIT 许可）改造而来。

| 文件 | 用途 |
| --- | --- |
| `favicon.svg` | 现代浏览器首选，矢量缩放不失真 |
| `favicon.ico` | 16/32/48 多尺寸（内置 PNG） |
| `favicon-16x16.png`、`favicon-32x32.png` | 标签页 / 书签栏 |
| `apple-touch-icon.png` | iOS/iPadOS 添加到主屏（直角满图，由系统自己裁圆角） |
| `android-chrome-192x192.png`、`android-chrome-512x512.png`、`mstile-150x150.png`、`site.webmanifest` | Android / Windows 磁贴 |
| `safari-pinned-tab.svg` | Safari 固定标签页（单色，颜色由 `params.favicon.mask` 决定） |

想换颜色就改 `static/favicon.svg` 里的 `fill="#1b1c1d"` / `stroke="#fe5186"`，其余 PNG/ICO 需要重新
栅格化（用无头 Chrome 按各尺寸导出即可，需要的话我可以帮你重新跑一遍）。

## 页脚与署名（哪些能删）

页脚显示什么完全由 `hugo.toml` 里 `[params.footer]` 的开关决定，**没有任何一项是模板强制要求的**：

| 字段 | 作用 | 当前值 |
| --- | --- | --- |
| `trademark` | `© 年份`，填数字即显示，删掉则不显示 | `2026` |
| `author` | 作者名链接，取 `params.author.name` | `true`（显示 Luke） |
| `rss` | RSS 图标（那个像 wifi 的图标） | `true` |
| `copyright` | 版权声明，取站点 `copyright` / `[languages.xx] copyright` | `false`（已关） |
| `topText` / `bottomText` | 页脚上下两行自定义 HTML | 均为空（已清空） |

RSS 订阅源里的 `<copyright>` 取站点顶层 `copyright`（现在是 `© 2026 Luke`），与页脚显示无关。

**关于主题署名是不是必须保留**：主题 `hello-friend-ng` 是 MIT 许可。MIT 的要求是「分发时保留版权声明和许可文本」，也就是仓库里的 `themes/hello-friend-ng/LICENSE.md` 必须留着（它已经在仓库里），**并不要求**在页面上显示 "Powered by Hugo / Theme ..."。所以把页脚那行删掉完全没问题。

## 主题来源

主题 `hello-friend-ng` 由 [Djordje Atlialp](https://github.com/rhazdon) 开发，基于 MIT License 发布，
许可证见 `themes/hello-friend-ng/LICENSE.md`。

## 分类

站点用 Hugo 的 taxonomy 划分两个方向，每个方向一个目录，目录名就是 URL 里的 key：

| key | 显示名 | 文件 |
| --- | --- | --- |
| `learning` | Tech Learning | `content/categories/learning/_index.md` |
| `engineering` | Engineering Practice | `content/categories/engineering/_index.md` |

- 给文章归类：在 front matter 里写 `categories: ["learning"]`，也可以同时写两个。
- 首页的「CATEGORIES」卡片自动来自 `content/categories/` 下的目录：顺序由各自 `_index.md` 的 `weight` 决定，描述取 `description`，文章数自动统计（草稿不计入）。
- 分类页地址是 `/categories/<key>/`，`/categories/` 是总览页；文章正文顶部的分类链接由主题的 `partials/categories.html` 渲染，不用改。
- 新增方向：建一个 `content/categories/<key>/_index.md`，写好 `title`、`description`、`weight` 就行，模板和配置都不用动；某个分类暂时没有文章也会显示卡片。
- key 用英文：中文 key 会生成 `/categories/%E6%8A%80%E6%9C%AF/` 这种百分号 URL。
- 首页「LATEST POSTS」区块显示最新 `homePostsCount`（当前 3）篇文章。