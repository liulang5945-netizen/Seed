import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import TrainingDatasetPanel from '../components/TrainingDatasetPanel.vue'

const t = (key) => ({
  train_upload: '上传训练数据',
  train_support: '支持格式',
  train_no_data: '暂无数据',
  dataset_preview: '数据集预览',
  samples: '样本',
  document: '文档',
}[key] || key)

const baseProps = {
  active: true,
  trainFiles: ['simple_zh/first.jsonl'],
  fileSizes: { 'simple_zh/first.jsonl': 2048 },
  selectedDatasets: [],
  trainPreview: {
    count: 1,
    native_trainable: true,
    report: { truncated: false },
    samples: [{ text: 'hello world' }],
  },
  allSelected: false,
  t,
}

describe('TrainingDatasetPanel', () => {
  it('renders dataset rows and preview while forwarding product events', async () => {
    const wrapper = mount(TrainingDatasetPanel, {
      props: baseProps,
      global: {
        stubs: {
          FileUploadQueue: { template: '<div class="upload-stub"></div>' },
          NButton: { template: '<button @click="$emit(\'click\')"><slot /></button>' },
          NCheckbox: { template: '<button class="checkbox-stub"><slot /></button>' },
          NEmpty: { template: '<div><slot /></div>' },
        },
      },
    })

    expect(wrapper.find('section.tab-panel').classes()).toContain('active')
    expect(wrapper.text()).toContain('simple_zh/first.jsonl')
    expect(wrapper.text()).toContain('数据集预览')
    expect(wrapper.text()).toContain('hello world')
    expect(wrapper.text()).toContain('2.0 KB')

    await wrapper.find('.dataset-toolbar button').trigger('click')
    await wrapper.find('.ds-act button').trigger('click')
    await wrapper.find('.ds-act button.danger').trigger('click')

    expect(wrapper.emitted('refresh')).toHaveLength(1)
    expect(wrapper.emitted('preview')).toEqual([['simple_zh/first.jsonl']])
    expect(wrapper.emitted('delete')).toEqual([['simple_zh/first.jsonl']])
  })

  it('keeps upload UI isolated from the parent data mutation owner', () => {
    const wrapper = mount(TrainingDatasetPanel, {
      props: { ...baseProps, trainFiles: [] },
      global: {
        stubs: {
          FileUploadQueue: { template: '<div class="upload-stub"></div>' },
          NEmpty: { props: ['description'], template: '<div class="empty-stub">{{ description }}</div>' },
        },
      },
    })

    expect(wrapper.find('.upload-stub').exists()).toBe(true)
    expect(wrapper.text()).toContain('暂无数据')
  })
})
