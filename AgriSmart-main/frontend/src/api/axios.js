import axios from "axios";

const api = axios.create({ 
  baseURL: process.env.REACT_APP_API_URL || "http://127.0.0.1:5000" 
});

api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("token");
    if (token) config.headers.Authorization = `Bearer ${token}`;
    return config;
  },
  (error) => Promise.reject(error)
);

api.interceptors.response.use(
  (resp) => resp,
  (error) => {
    try {
      const status = error.response?.status;
      if (status === 401) {
        localStorage.removeItem("token");
        localStorage.removeItem("user");
        setTimeout(() => (window.location.href = "/login"), 250);
      }
    } catch (e) {}
    return Promise.reject(error);
  }
);

export default api;
