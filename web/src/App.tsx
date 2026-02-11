import { BrowserRouter, Routes, Route } from 'react-router-dom';
import PuzzleSelector from './components/PuzzleSelector';
import CrosswordPlayer from './components/CrosswordPlayer';
import UploadPage from './components/UploadPage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<PuzzleSelector />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/puzzle/:id" element={<CrosswordPlayer />} />
      </Routes>
    </BrowserRouter>
  );
}
