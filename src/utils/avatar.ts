// Place your avatar image in the public folder as 'avatar.jpg' or 'avatar.png'
// This function will return the correct path

export const getAvatarUrl = (): string => {
  // Check if avatar exists in public folder
  // You can place your photo as /public/avatar.jpg or /public/avatar.png

  // Try common image formats
  const formats = ['jpg', 'jpeg', 'png', 'webp'];

  // For now, return the path - the browser will handle if it exists
  // You should place your image in public/avatar.jpg
  return '/avatar.jpg';
};

// Alternative: Use imported image
// If you want to import the image directly, place it in src/assets/
// and uncomment below:

// import avatarImage from '../assets/avatar.jpg';
// export const getAvatarUrl = () => avatarImage;
