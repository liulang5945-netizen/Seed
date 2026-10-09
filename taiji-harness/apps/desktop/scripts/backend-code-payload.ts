/** Read the Taiji base the shipped runtime will load from the source that ships beside it. */

/**
 * Name the checkpoint the runtime loads when a request names none.
 *
 * The payload has to carry the file `SeedRuntime` resolves on its own, so the name is
 * read out of `api/seed_runtime.py` instead of copied here. A source this reader cannot
 * parse, or one that names more than one file, fails the build rather than shipping a
 * base the running backend will not load.
 * @param seedRuntimeSource - text of the packaged `api/seed_runtime.py`.
 * @returns file name of the product's default checkpoint.
 */
export function defaultCheckpointName(seedRuntimeSource: string): string {
  const assignment = /^DEFAULT_CHECKPOINT = \((?<wrapped>[\s\S]*?)^\)|^DEFAULT_CHECKPOINT = (?<inline>.*)$/mu
    .exec(seedRuntimeSource)
  if (assignment === null) throw new Error('desktop backend: api/seed_runtime.py has no DEFAULT_CHECKPOINT assignment')
  const names = [...(assignment.groups?.wrapped ?? assignment.groups?.inline ?? '').matchAll(/"([^"]+\.pt)"/gu)]
    .map(match => match[1])
  const [name] = names
  if (names.length !== 1 || name === undefined) {
    throw new Error(`desktop backend: DEFAULT_CHECKPOINT must name one .pt file, found ${String(names.length)}`)
  }
  return name
}
