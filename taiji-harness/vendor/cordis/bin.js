#!/usr/bin/env node

import { Context } from '@taiji/cordis'
import { pathToFileURL } from 'node:url'
import Loader from '@taiji/cordis-plugin-loader'

const ctx = new Context()
ctx.baseUrl = pathToFileURL(process.cwd()).href + '/'

await ctx.plugin(Loader)
await ctx.loader.create({
  name: '@taiji/cordis-plugin-include',
  config: {
    path: './cordis.yml',
  },
})
