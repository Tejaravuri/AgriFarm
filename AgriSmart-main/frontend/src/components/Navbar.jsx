import { Link, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { useState } from "react";

export default function Navbar() {
  const { t, i18n } = useTranslation();
  const navigate = useNavigate();

  const [langOpen, setLangOpen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  const isLoggedIn = localStorage.getItem("token");

  const changeLanguage = (lang) => {
    i18n.changeLanguage(lang);
    setLangOpen(false);
    setMenuOpen(false);
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    navigate("/");
    setMenuOpen(false);
  };

  return (
    <nav className="w-full bg-[#0b1c2d] text-white">
      <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">

        {/* LOGO */}
        <Link to="/" className="flex items-center text-2xl font-bold">
          AgriSmart <span className="ml-1">🌱</span>
        </Link>

        {/* DESKTOP NAV */}
        <div className="hidden md:flex items-center gap-8 text-sm font-medium">

          {/* LANGUAGE */}
          <div className="relative">
            <button
              onClick={() => setLangOpen(!langOpen)}
              className="flex items-center gap-1 hover:text-green-400 transition"
            >
              🌐 <span className="uppercase">{i18n.language}</span>
            </button>

            {langOpen && (
              <div className="absolute right-0 mt-3 w-36 bg-white text-gray-800 rounded-lg shadow-lg overflow-hidden z-50">
                {[
                  ["en", "English"],
                  ["te", "తెలుగు"],
                  ["hi", "हिंदी"],
                  ["ta", "தமிழ்"],
                  ["kn", "ಕನ್ನಡ"],
                ].map(([code, label]) => (
                  <button
                    key={code}
                    onClick={() => changeLanguage(code)}
                    className="w-full px-4 py-2 hover:bg-gray-100 text-left"
                  >
                    {label}
                  </button>
                ))}
              </div>
            )}
          </div>

          <Link to="/" className="hover:text-green-400 transition">
            {t("navbar.home")}
          </Link>

          <Link to="/help" className="hover:text-green-400 transition">
            {t("navbar.help")}
          </Link>

          {!isLoggedIn && (
            <>
              <Link to="/login" className="hover:text-green-400 transition">
                {t("navbar.login")}
              </Link>

              <Link
                to="/signup"
                className="bg-green-500 px-5 py-2 rounded-md hover:bg-green-600 transition"
              >
                {t("navbar.signup")}
              </Link>
            </>
          )}

          {isLoggedIn && (
            <button
              onClick={handleLogout}
              className="bg-red-500 px-5 py-2 rounded-md hover:bg-red-600 transition"
            >
              {t("navbar.logout")}
            </button>
          )}
        </div>

        {/* MOBILE BUTTON */}
        <button
          className="md:hidden text-2xl"
          onClick={() => setMenuOpen(!menuOpen)}
        >
          {menuOpen ? "✕" : "☰"}
        </button>
      </div>

      {/* MOBILE MENU */}
      {/* MOBILE MENU */}
{menuOpen && (
  <div className="md:hidden bg-[#0b1c2d] px-6 py-6 space-y-6 text-sm">

    {/* LANGUAGE */}
    <div>
      <p className="text-gray-400 mb-3">Language</p>
      <div className="flex flex-wrap gap-3">
        {["en", "te", "hi", "ta", "kn"].map((lng) => (
          <button
            key={lng}
            onClick={() => changeLanguage(lng)}
            className={`px-4 py-1.5 rounded-md border border-white/30 ${
              i18n.language === lng ? "bg-green-500 text-white" : ""
            }`}
          >
            {lng.toUpperCase()}
          </button>
        ))}
      </div>
    </div>

    <div className="border-t border-white/20" />

    {/* NAV LINKS */}
    <div className="flex flex-col items-center gap-5 text-base">
      <Link onClick={() => setMenuOpen(false)} to="/">
        {t("navbar.home")}
      </Link>

      <Link onClick={() => setMenuOpen(false)} to="/help">
        {t("navbar.help")}
      </Link>

      {!isLoggedIn && (
        <Link onClick={() => setMenuOpen(false)} to="/login">
          {t("navbar.login")}
        </Link>
      )}
    </div>

    {!isLoggedIn && (
      <Link
        onClick={() => setMenuOpen(false)}
        to="/signup"
        className="block bg-green-500 text-center py-3 rounded-lg text-base font-semibold"
      >
        {t("navbar.signup")}
      </Link>
    )}

    {isLoggedIn && (
      <button
        onClick={handleLogout}
        className="w-full bg-red-500 py-3 rounded-lg text-base font-semibold"
      >
        {t("navbar.logout")}
      </button>
    )}
  </div>
)}
    </nav>
  );
}
