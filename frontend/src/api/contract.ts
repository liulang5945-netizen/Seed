/**
 * 编译期 API 契约门 —— 前端侧唯一的 OpenAPI 快照消费点。
 *
 * 为什么需要它：`scripts/check-api-contract.mjs` 用 227 行手写括号匹配，从 JS
 * 源码里抽出 `JSON.stringify({...})` 的字段名再与快照比对；那是文本级事后检查。
 * 端点被后端改名、或 facade 声明了根本不存在的路径，只有运行到那一步才暴露成 404。
 * 本文件把同一件事提到编译期：`vue-tsc --noEmit` 直接把漂移报成类型错误。
 *
 * 单一真源链路（三段都可审计）：
 *   tests/snapshots/openapi_baseline.json     ← 冻结快照，已由后端 CI 守护
 *     → npm run gen:api-types                 → ./schema.d.ts（提交入库）
 *     → npm run check:api-types               → 生成物漂移即红
 *     → ./contract.ts（本文件）                → 路径与 payload 的编译期门
 *
 * 失败时的读法：若下面某个 AssertNever 报 “does not satisfy the constraint
 * 'never'”，报错信息里列出的字面量就是越界项 —— 要么后端改了契约，要么
 * facade 写错了路径，先确认是哪一侧再改。
 */

import type { components, paths } from './schema'
import { nativeApiPaths } from './paths'

/** 快照中全部端点路径的字面量联合。 */
export type ApiPath = keyof paths & string

/** 快照 components.schemas 的别名入口，请求/响应体类型从这里取。 */
export type ApiSchemas = components['schemas']
export type ApiSchema<K extends keyof ApiSchemas> = ApiSchemas[K]

/** 把 facade 的两层路径表压平成叶子字面量联合。 */
type PathLeaves<T> = {
  [K in keyof T]: T[K] extends Record<string, unknown> ? T[K][keyof T[K]] : T[K]
}[keyof T]

type DeclaredPath = Extract<PathLeaves<typeof nativeApiPaths>, string>

type AssertNever<T extends never> = T

/**
 * 正向钉住：facade 声明的每一个路径都必须存在于快照。
 * 后端删掉/改写了某个端点而前端未同步 ⇒ 此处编译失败。
 */
export type FacadePathsExistInSnapshot = AssertNever<Exclude<DeclaredPath, ApiPath>>

/**
 * 端点操作类型，以及按 operation 取 JSON 请求体 / 200 响应体的工具类型。
 * 下一步把 facade 的 `payload: Object` 换成这些类型时直接复用，不需要再解析快照。
 */
export type OperationOf<P extends ApiPath, M extends keyof paths[P]> = paths[P][M]

export type JsonRequestBodyOf<P extends ApiPath, M extends keyof paths[P]> =
  OperationOf<P, M> extends { requestBody: { content: { 'application/json': infer Body } } }
    ? Body
    : never

export type JsonResponseOf<P extends ApiPath, M extends keyof paths[P]> =
  OperationOf<P, M> extends { responses: { 200: { content: { 'application/json': infer Result } } } }
    ? Result
    : never
