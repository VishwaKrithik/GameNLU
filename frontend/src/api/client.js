import axios from 'axios'

const api = axios.create({
  baseURL: 'https://gamenlu.onrender.com/',
})

export const predictCommands = async (payload) => {
  const response = await api.post('/predict', payload)
  return response.data
}

export default api
