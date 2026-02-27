import Navbar from "../components/Navbar";
import { Link, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { useRef } from "react";

export default function Home() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const cardsRef = useRef(null);

  const isLoggedIn = localStorage.getItem("token");

  const scrollToCards = () => {
    cardsRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleGetStarted = () => {
    if (!isLoggedIn) {
      navigate("/login");
    } else {
      scrollToCards();
    }
  };

  return (
    <>
      <Navbar />

      
      <section
        className="relative h-[80vh] bg-cover bg-center"
        style={{ backgroundImage: "url('/images/crop1.jpg')" }}
      >
        <div className="absolute inset-0 bg-black/35"></div>

        <div className="relative z-10 max-w-7xl mx-auto px-8 pt-32 text-white">
          <h1 className="text-4xl md:text-5xl font-bold leading-tight mb-4">
            {t("home.heroTitle")}
          </h1>

          <p className="max-w-2xl text-lg text-gray-200">
            {t("home.heroSubtitle")}
          </p>

          <div className="mt-8 flex gap-4">
            <button
              onClick={handleGetStarted}
              className="bg-green-500 hover:bg-green-600 px-6 py-3 rounded-md font-medium transition"
            >
              {t("common.getStarted")}
            </button>

            <button
              onClick={scrollToCards}
              className="bg-white text-gray-900 px-6 py-3 rounded-md font-medium hover:bg-gray-100 transition"
            >
              {t("common.learnMore")}
            </button>
          </div>
        </div>
      </section>

      
      <section
        ref={cardsRef}
        className="py-20"
        style={{
          background:
            "linear-gradient(rgba(240,253,244,0.95), rgba(240,253,244,0.95)), url('/images/crop.jpg')",
          backgroundSize: "cover",
        }}
      >
        <div className="text-center mb-14">
          <h2 className="text-3xl font-bold text-green-800">
            {t("home.platformTitle")}
          </h2>
          <p className="text-gray-700 mt-2">
            {t("home.platformSubtitle")}
          </p>
        </div>

        <div className="max-w-7xl mx-auto px-10 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-10">
          {[
            {
              title: t("modules.crop.title"),
              desc: t("modules.crop.desc"),
              img: "/images/crop.jpg",
              link: "/crop",
            },
            {
              title: t("modules.fertilizer.title"),
              desc: t("modules.fertilizer.desc"),
              img: "/images/fertilizer.jpg",
              link: "/fertilizer",
            },
            {
              title: t("modules.disease.title"),
              desc: t("modules.disease.desc"),
              img: "/images/disease.jpg",
              link: "/disease",
            },
            {
              title: t("modules.pest.title"),
              desc: t("modules.pest.desc"),
              img: "/images/pest.jpg",
              link: "/pest",
            },
            {
              title: t("modules.yield.title"),
              desc: t("modules.yield.desc"),
              img: "/images/yield.jpg",
              link: "/yield",
            },
          ].map((item, index) => (
            <div
              key={index}
              className="bg-white rounded-xl shadow-lg transition transform hover:-translate-y-2 hover:shadow-2xl"
            >
              <img
                src={item.img}
                alt={item.title}
                className="h-48 w-full object-cover rounded-t-xl"
              />
              <div className="p-6">
                <h3 className="text-xl font-semibold mb-2">
                  {item.title}
                </h3>
                <p className="text-gray-600 mb-4">
                  {item.desc}
                </p>
                <Link
                  to={item.link}
                  className="block text-center bg-green-600 text-white py-2 rounded-md hover:bg-green-700"
                >
                  {t("common.explore")}
                </Link>
              </div>
            </div>
          ))}
        </div>
      </section>

      
      <footer className="bg-slate-900 text-gray-400 py-10">
        <div className="max-w-7xl mx-auto px-8 text-center">
          <h3 className="text-white text-xl font-semibold mb-2">
            AgriSmart 🌱
          </h3>
          <p className="text-sm">
            {t("footer.tagline")}
          </p>
          <p className="text-xs mt-4">
            © {new Date().getFullYear()} AgriSmart. {t("footer.rights")}
          </p>
        </div>
      </footer>
    </>
  );
}
