import { useState } from "react";
import { useTranslation } from "react-i18next";
import api from "../api/axios";
import ResultCard from "../components/ResultCard";

export default function Fertilizer() {
  const { t } = useTranslation();
  const [form, setForm] = useState({
    crop: "",
    temperature: "",
    humidity: "",
    moisture: "",
    soil_type: "",
    nitrogen: "",
    phosphorous: "",
    potassium: "",
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) =>
    setForm({ ...form, [e.target.name]: e.target.value });

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    const token = localStorage.getItem("token");
    if (!token) {
      setError("Please login first");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const dataToSend = {
        crop: form.crop,
        temperature: form.temperature ? parseFloat(form.temperature) : 25,
        humidity: form.humidity ? parseFloat(form.humidity) : 50,
        moisture: form.moisture ? parseFloat(form.moisture) : 40,
        soil_type: form.soil_type || "Loamy",
        nitrogen: form.nitrogen ? parseFloat(form.nitrogen) : 0,
        phosphorous: form.phosphorous ? parseFloat(form.phosphorous) : 0,
        potassium: form.potassium ? parseFloat(form.potassium) : 0
      };
      const res = await api.post("/predict-fertilizer", dataToSend);
      setResult(res.data);
    } catch (err) {
      console.error("Fertilizer request error:", err);
      const status = err.response?.status;
      const serverMsg = err.response?.data || err.message;
      setError(
        status ? `Error ${status}: ${JSON.stringify(serverMsg)}` : String(serverMsg || "Fertilizer prediction failed")
      );
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen flex bg-gray-100">
      <div className="hidden md:block w-1/2">
        <img src="/images/fertilizer.jpg" alt="Fertilizer Recommendation" className="h-full w-300 object-cover" />
      </div>

      <div className="w-full md:w-1/2 flex items-center justify-center p-10">
        <div className="bg-white p-8 rounded-xl shadow-xl w-full max-w-lg">
          <h2 className="text-2xl font-bold text-green-700 mb-6">
            {t('modules.fertilizer.title')}
          </h2>

          <form onSubmit={handleSubmit} className="space-y-4">
            {Object.keys(form).map((key) => {
              const placeholders = {
                crop: t('modules.fertilizer.cropType'),
                temperature: t('modules.crop.temperature'),
                humidity: t('modules.crop.humidity'),
                moisture: t('modules.fertilizer.moisture'),
                soil_type: t('modules.fertilizer.soilType'),
                nitrogen: t('modules.crop.nitrogen'),
                phosphorous: t('modules.crop.phosphorus'),
                potassium: t('modules.crop.potassium')
              };
              
              if (key === "soil_type") {
                return (
                  <select
                    key={key}
                    name={key}
                    value={form[key]}
                    onChange={handleChange}
                    className="w-full border p-3 rounded"
                  >
                    <option value="">{t('modules.fertilizer.selectSoil')}</option>
                    <option value="Sandy">Sandy</option>
                    <option value="Loamy">Loamy</option>
                    <option value="Black">Black</option>
                    <option value="Red">Red</option>
                  </select>
                );
              }
              
              return (
                <input
                  key={key}
                  name={key}
                  placeholder={placeholders[key]}
                  type={key === "crop" ? "text" : "number"}
                  step="0.01"
                  onChange={handleChange}
                  required={key === "crop"}
                  className="w-full border p-3 rounded"
                />
              );
            })}

            <button 
              type="submit"
              disabled={loading}
              className="w-full bg-green-600 text-white py-3 rounded hover:bg-green-700 disabled:bg-gray-400"
            >
              {loading ? t('common.loading') : t('modules.fertilizer.submit')}
            </button>
          </form>

          {error && (
            <p className="mt-4 p-3 bg-red-100 text-red-700 rounded">
              ❌ {error}
            </p>
          )}

          {result && (
            <ResultCard
              title="Fertilizer Recommendation"
              subtitle="Best match for your conditions"
              main={<span className="text-green-700 font-bold">{result.fertilizer}</span>}
              bullets={result.recommendations || []}
              accent="green"
            />
          )}
        </div>
      </div>
    </div>
  );
}
