import { useState } from "react";
import { useTranslation } from "react-i18next";
import api from "../api/axios";
import ResultCard from "../components/ResultCard";

export default function Disease() {
  const { t } = useTranslation();
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please select an image");
      return;
    }
    
    const token = localStorage.getItem("token");
    if (!token) {
      setError("Please login first");
      return;
    }

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append("image", file);

    try {
      const res = await api.post("/predict-disease", formData);
      setResult(res.data);
    } catch (err) {
      console.error("Disease request error:", err);
      const status = err.response?.status;
      const serverMsg = err.response?.data || err.message;
      setError(
        status ? `Error ${status}: ${JSON.stringify(serverMsg)}` : String(serverMsg || "Disease detection failed")
      );
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen flex bg-gray-100">
      <div className="hidden md:block w-1/2">
        <img src="/images/disease1.jpg" alt="Disease Detection" className="h-full w-full object-cover" />
      </div>

      <div className="w-full md:w-1/2 flex items-center justify-center p-10">
        <div className="bg-white p-8 rounded-xl shadow-xl w-full max-w-lg">
          <h2 className="text-2xl font-bold text-green-700 mb-6">
            {t('modules.disease.title')}
          </h2>

          <form onSubmit={handleSubmit} className="space-y-4">
            <input
              type="file"
              accept="image/*"
              onChange={(e) => setFile(e.target.files[0])}
              required
              className="w-full border p-3 rounded"
            />

            <button 
              type="submit"
              disabled={loading}
              className="w-full bg-green-600 text-white py-3 rounded hover:bg-green-700 disabled:bg-gray-400"
            >
              {loading ? t('common.loading') : t('modules.disease.submit')}
            </button>
          </form>

          {error && (
            <p className="mt-4 p-3 bg-red-100 text-red-700 rounded">
              ❌ {error}
            </p>
          )}

          {result && (
            result.message ? (
              <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <p className="text-blue-900 text-sm font-semibold mb-2">📋 Detection Result</p>
                <p className="text-blue-800">{result.message}</p>
              </div>
            ) : (
              <ResultCard
                title="Disease Detection"
                subtitle={result.disease}
                main={<span className="text-red-600 font-bold">{result.disease}</span>}
                confidence={result.confidence}
                imageUrl={file ? URL.createObjectURL(file) : null}
                bullets={result.recommendations || []}
                accent="yellow"
              />
            )
          )}
        </div>
      </div>
    </div>
  );
}
