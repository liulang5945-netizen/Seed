/**
 * 让 jsdom 的 document.createElement 与 Blink 一样严格地校验标签名。
 *
 * 为什么需要这个文件：jsdom 的 createElement 不校验标签名合法性，
 * `document.createElement('📤')` 会安然返回一个 tagName 为 emoji 的元素；
 * 而 Blink（QtWebEngine / Chrome）会同步抛 InvalidCharacterError。
 *
 * Vue 的 `<component :is="x">` 在 x 为字符串且解析不到组件时，会退化成原生标签
 * 交给 createElement。于是「把 emoji 当图标名传进去」这类错误在 jsdom 下静默通过，
 * 在真实客户端里却在渲染期抛异常、炸掉整棵 router-view 子树 —— 表现为点进某个
 * 页面后所有标签页全白屏，而 181 个单测依然全绿。这是已经付过学费的失效模式，
 * 必须由环境门禁而不是逐组件补测试来兜住。
 *
 * 校验规则对齐 HTML 规范的 isValidElementName：首字符必须是 ASCII 字母，
 * 后续字符不得包含 NUL / 空白 / '/' / '>'。
 */
const VALID_ELEMENT_NAME = /^[A-Za-z][^\0\t\n\f\r >/]*$/;

const nativeCreateElement = Document.prototype.createElement;

Document.prototype.createElement = function createElement(tagName, options) {
  if (!VALID_ELEMENT_NAME.test(String(tagName))) {
    throw new DOMException(
      `Failed to execute 'createElement' on 'Document': The tag name provided ('${tagName}') is not a valid name.`,
      'InvalidCharacterError'
    );
  }
  return nativeCreateElement.call(this, tagName, options);
};
