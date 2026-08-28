/**
 * 认证管理 composable
 * 提供登录、登出、Token 管理、自动附加认证头等功能
 *
 * 认证状态（authEnabled、username）由 runtimeStore 统一管理，
 * 此处只提供命令动作（login/logout/enable/disable）。
 */
import { ref, computed } from 'vue';
import { nativeApi } from './nativeApi.js';
import { useRuntimeStore } from '@/stores/runtimeStore.js';

const token = ref(localStorage.getItem('jwt_token') || '');

export function useAuth() {
  const runtimeStore = useRuntimeStore()

  // 从 runtimeStore 读取认证状态，不再自己轮询 /api/auth/status
  const authEnabled = computed(() => runtimeStore.auth?.enabled ?? false)
  const username = computed(() => runtimeStore.auth?.username ?? '')
  const authLoaded = computed(() => !!runtimeStore.runtimeSnapshot)
  const isAuthenticated = computed(() => !authEnabled.value || !!token.value);

  async function login(user, password) {
    const d = await nativeApi.authLogin({ username: user, password });
    token.value = d.token;
    localStorage.setItem('jwt_token', d.token);
    return d;
  }

  function logout() {
    token.value = '';
    localStorage.removeItem('jwt_token');
  }

  function getAuthHeaders() {
    const headers = { 'Content-Type': 'application/json' };
    if (token.value) {
      headers['Authorization'] = `Bearer ${token.value}`;
    }
    return headers;
  }

  async function enableAuth(user, password) {
    try {
      await nativeApi.authEnable({ username: user, password });
      runtimeStore.auth.enabled = true;
      runtimeStore.auth.username = user;
      return true;
    } catch (e) {
      return false;
    }
  }

  async function disableAuth() {
    try {
      await nativeApi.authDisable();
      runtimeStore.auth.enabled = false;
      logout();
      return true;
    } catch (e) {
      return false;
    }
  }

  async function changePassword(oldPwd, newPwd) {
    try {
      await nativeApi.authChangePassword({ old_password: oldPwd, new_password: newPwd });
      return true;
    } catch (e) {
      return false;
    }
  }

  // 处理 401 响应（全局拦截）
  function handleAuthError(response) {
    if (response.status === 401) {
      logout();
      return true;
    }
    return false;
  }

  return {
    token,
    authEnabled,
    username,
    authLoaded,
    isAuthenticated,
    login,
    logout,
    getAuthHeaders,
    enableAuth,
    disableAuth,
    changePassword,
    handleAuthError,
  };
}
