import type { UploadResponse } from '../types/api';

interface GridPreviewProps {
  data: UploadResponse;
  onConfirm: () => void;
  onReupload: () => void;
}

export default function GridPreview({ data, onConfirm, onReupload }: GridPreviewProps) {
  return (
    <div style={styles.container}>
      <h2 style={styles.title}>Grid Detected</h2>
      <div style={styles.imageWrap}>
        <img
          src={data.warped_grid_url}
          alt="Detected crossword grid"
          style={styles.image}
        />
      </div>
      <div style={styles.stats}>
        <span>Grid: {data.grid_size[0]} &times; {data.grid_size[1]}</span>
        <span>{data.clue_slot_count} clue slots</span>
      </div>
      <div style={styles.actions}>
        <button style={styles.secondaryButton} onClick={onReupload}>
          Re-upload
        </button>
        <button style={styles.primaryButton} onClick={onConfirm}>
          Edit Grid &rarr;
        </button>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '16px',
  },
  title: {
    margin: 0,
    fontSize: '20px',
    fontWeight: 600,
  },
  imageWrap: {
    border: '1px solid #ddd',
    borderRadius: '8px',
    overflow: 'hidden',
    maxWidth: '500px',
  },
  image: {
    display: 'block',
    width: '100%',
    height: 'auto',
  },
  stats: {
    display: 'flex',
    gap: '24px',
    fontSize: '15px',
    color: '#555',
  },
  actions: {
    display: 'flex',
    gap: '12px',
    marginTop: '8px',
  },
  primaryButton: {
    padding: '10px 24px',
    backgroundColor: '#2563eb',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    fontSize: '15px',
    cursor: 'pointer',
  },
  secondaryButton: {
    padding: '10px 24px',
    backgroundColor: '#fff',
    color: '#555',
    border: '1px solid #ccc',
    borderRadius: '6px',
    fontSize: '15px',
    cursor: 'pointer',
  },
};
