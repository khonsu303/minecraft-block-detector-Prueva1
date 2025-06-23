import React, { useState } from 'react';
import {
  Container,
  Box,
  Typography,
  Paper,
  Button,
  CircularProgress,
  List,
  ListItem,
  ListItemText,
  Card,
  CardMedia,
  CardContent,
  ThemeProvider,
  createTheme,
  CssBaseline,
  alpha,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

// Crear un tema personalizado
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#4CAF50', // Verde Minecraft
      light: '#81C784',
      dark: '#388E3C',
    },
    secondary: {
      main: '#FFA000', // Naranja Minecraft
      light: '#FFB74D',
      dark: '#F57C00',
    },
    background: {
      default: '#1a1a1a',
      paper: '#2d2d2d',
    },
  },
  typography: {
    fontFamily: '"Minecraft", "Roboto", "Helvetica", "Arial", sans-serif',
    h3: {
      fontWeight: 700,
      textShadow: '2px 2px 4px rgba(0,0,0,0.5)',
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          padding: '10px 20px',
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: '0 6px 8px rgba(0,0,0,0.2)',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 8px 16px rgba(0,0,0,0.2)',
          transition: 'transform 0.3s ease-in-out',
          '&:hover': {
            transform: 'scale(1.02)',
          },
        },
      },
    },
  },
});

const VisuallyHiddenInput = styled('input')`
  clip: rect(0 0 0 0);
  clip-path: inset(50%);
  height: 1px;
  overflow: hidden;
  position: absolute;
  bottom: 0;
  left: 0;
  white-space: nowrap;
  width: 1px;
`;

const GradientBackground = styled(Box)(({ theme }) => ({
  minHeight: '100vh',
  background: `linear-gradient(135deg, ${alpha(theme.palette.primary.dark, 0.2)} 0%, ${alpha(theme.palette.background.default, 0.95)} 100%)`,
  padding: theme.spacing(4),
}));

const StyledPaper = styled(Paper)(({ theme }) => ({
  background: `linear-gradient(145deg, ${alpha(theme.palette.background.paper, 0.9)}, ${alpha(theme.palette.background.paper, 0.7)})`,
  backdropFilter: 'blur(10px)',
  borderRadius: 16,
  padding: theme.spacing(4),
  boxShadow: '0 8px 32px rgba(0,0,0,0.2)',
}));

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [training, setTraining] = useState(false);
  const [trainProgress, setTrainProgress] = useState([]);

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(URL.createObjectURL(file));
      setLoading(true);
      setError(null);
      setPredictions(null);

      const formData = new FormData();
      formData.append('image', file);

      try {
        const response = await fetch('http://localhost:5000/predict', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error('Error al procesar la imagen');
        }

        const data = await response.json();
        setPredictions(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
  };

  const handleTrainModel = () => {
    setTraining(true);
    setTrainProgress([]);
    const eventSource = new EventSource('http://localhost:5000/train');
    eventSource.onmessage = (event) => {
      setTrainProgress(prev => [...prev, event.data]);
      if (event.data === 'Entrenamiento finalizado') {
        setTraining(false);
        eventSource.close();
      }
    };
    eventSource.onerror = () => {
      setTraining(false);
      setTrainProgress(prev => [...prev, 'Error en el entrenamiento']);
      eventSource.close();
    };
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <GradientBackground>
        <Container maxWidth="md">
          <Box sx={{ my: 4, textAlign: 'center' }}>
            <Typography 
              variant="h3" 
              component="h1" 
              gutterBottom
              sx={{
                background: `linear-gradient(45deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                color: 'transparent',
                mb: 4,
              }}
            >
              Minecraft Block Detector
            </Typography>
            <Typography 
              variant="subtitle1" 
              color="text.secondary" 
              paragraph
              sx={{ mb: 4 }}
            >
              Sube una imagen de un bloque de Minecraft y te diré qué tipo de bloque es
            </Typography>

            <StyledPaper>
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3 }}>
                <Button
                  component="label"
                  variant="contained"
                  size="large"
                  startIcon={<CloudUploadIcon />}
                  disabled={loading}
                  sx={{
                    background: `linear-gradient(45deg, ${theme.palette.primary.main}, ${theme.palette.primary.light})`,
                    '&:hover': {
                      background: `linear-gradient(45deg, ${theme.palette.primary.dark}, ${theme.palette.primary.main})`,
                    },
                  }}
                >
                  Subir Imagen
                  <VisuallyHiddenInput type="file" onChange={handleImageUpload} accept="image/*" />
                </Button>

                {selectedImage && (
                  <Card sx={{ maxWidth: 345, mt: 2, width: '100%' }}>
                    <CardMedia
                      component="img"
                      height="300"
                      image={selectedImage}
                      alt="Bloque de Minecraft"
                      sx={{ 
                        objectFit: 'contain',
                        backgroundColor: alpha(theme.palette.background.paper, 0.5),
                      }}
                    />
                  </Card>
                )}

                {loading && (
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, my: 2 }}>
                    <CircularProgress color="primary" />
                    <Typography variant="h6">Analizando imagen...</Typography>
                  </Box>
                )}

                {error && (
                  <Typography 
                    color="error" 
                    sx={{ 
                      mt: 2,
                      p: 2,
                      borderRadius: 2,
                      backgroundColor: alpha(theme.palette.error.main, 0.1),
                    }}
                  >
                    {error}
                  </Typography>
                )}

                {predictions && (
                  <Card sx={{ width: '100%', mt: 2 }}>
                    <CardContent>
                      <Typography 
                        variant="h5" 
                        gutterBottom
                        sx={{
                          color: theme.palette.primary.main,
                          mb: 3,
                        }}
                      >
                        Resultados
                      </Typography>
                      <List>
                        {predictions.top_3_predictions.map((prediction, index) => (
                          <ListItem 
                            key={index}
                            sx={{
                              mb: 1,
                              borderRadius: 2,
                              backgroundColor: alpha(theme.palette.background.paper, 0.5),
                              '&:hover': {
                                backgroundColor: alpha(theme.palette.background.paper, 0.8),
                              },
                            }}
                          >
                            <ListItemText
                              primary={
                                <Typography variant="h6" color="primary">
                                  {prediction[0]}
                                </Typography>
                              }
                              secondary={
                                <Typography variant="body2" color="text.secondary">
                                  Confianza: {(prediction[1] * 100).toFixed(2)}%
                                </Typography>
                              }
                            />
                          </ListItem>
                        ))}
                      </List>
                    </CardContent>
                  </Card>
                )}

                <Button
                  variant="contained"
                  color="secondary"
                  onClick={handleTrainModel}
                  disabled={training}
                  sx={{ mb: 3 }}
                >
                  {training ? 'Entrenando...' : 'Entrenar modelo'}
                </Button>
                {trainProgress.length > 0 && (
                  <Paper sx={{ p: 2, mb: 3, backgroundColor: 'rgba(255,255,255,0.05)' }}>
                    <Typography variant="h6" color="secondary" gutterBottom>
                      Progreso del entrenamiento
                    </Typography>
                    <Box sx={{ maxHeight: 200, overflowY: 'auto', textAlign: 'left' }}>
                      {trainProgress.map((msg, idx) => (
                        <Typography key={idx} variant="body2" sx={{ color: msg.includes('finalizado') ? 'green' : 'inherit' }}>
                          {msg}
                        </Typography>
                      ))}
                    </Box>
                  </Paper>
                )}
              </Box>
            </StyledPaper>
          </Box>
        </Container>
      </GradientBackground>
    </ThemeProvider>
  );
}

export default App; 