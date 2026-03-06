import { BrowserRouter, Routes, Route } from 'react-router-dom';
import PuzzleSelector from './components/PuzzleSelector';
import CrosswordPlayer from './components/CrosswordPlayer';
import UploadPage from './components/UploadPage';
import MaskRetryPage from './components/MaskRetryPage';
import { ThemeProvider, useTheme } from './hooks/useTheme';

function ThemeToggle() {
  const { dark, toggle } = useTheme();
  return (
    <button
      onClick={toggle}
      className="theme-toggle"
      style={{
        position: 'fixed',
        top: 12,
        right: 12,
        zIndex: 3000,
        background: 'none',
        border: '1px solid var(--border)',
        borderRadius: '50%',
        width: 36,
        height: 36,
        fontSize: 18,
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: 'var(--text-secondary)',
        backgroundColor: 'var(--bg-secondary)',
      }}
      title={dark ? 'Switch to light mode' : 'Switch to dark mode'}
    >
      {dark ? '\u2600\uFE0F' : '\uD83C\uDF19'}
    </button>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <ThemeToggle />
        <Routes>
          <Route path="/" element={<PuzzleSelector />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/puzzle/:id" element={<CrosswordPlayer />} />
          <Route path="/mask/:sessionId" element={<MaskRetryPage />} />
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
}
