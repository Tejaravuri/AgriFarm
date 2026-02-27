import Navbar from "../components/Navbar";
import { useTranslation } from "react-i18next";

export default function Help() {
  const { t } = useTranslation();

  return (
    <>
      <Navbar />

      <div className="min-h-screen bg-gray-100 p-8">
        <div className="max-w-5xl mx-auto bg-white p-10 rounded-xl shadow">

          <h1 className="text-3xl font-bold text-green-700 mb-6">
            {t('help.title')}
          </h1>

          <p className="text-gray-700 mb-8">
            {t('help.intro')}
          </p>

          <div className="space-y-6">

            <div>
              <h2 className="text-xl font-semibold text-green-600">
                {t('help.crop.title')}
              </h2>
              <p>
                {t('help.crop.desc')}
              </p>
            </div>

            <div>
              <h2 className="text-xl font-semibold text-green-600">
                {t('help.fertilizer.title')}
              </h2>
              <p>
                {t('help.fertilizer.desc')}
              </p>
            </div>

            <div>
              <h2 className="text-xl font-semibold text-green-600">
                {t('help.disease.title')}
              </h2>
              <p>
                {t('help.disease.desc')}
              </p>
            </div>

            <div>
              <h2 className="text-xl font-semibold text-green-600">
                {t('help.pest.title')}
              </h2>
              <p>
                {t('help.pest.desc')}
              </p>
            </div>

            <div>
              <h2 className="text-xl font-semibold text-green-600">
                {t('help.yield.title')}
              </h2>
              <p>
                {t('help.yield.desc')}
              </p>
            </div>

          </div>

          <div className="mt-10 bg-green-100 p-4 rounded text-green-800">
            {t('help.tip')}
          </div>

        </div>
      </div>
    </>
  );
}
