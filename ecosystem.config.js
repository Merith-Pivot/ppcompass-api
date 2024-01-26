module.exports = {
  apps: [
    {
      name: 'ppc-api',
      script: 'uvicorn',
      args: 'app.main:app --reload',
      watch: true,
      env: {
        NODE_ENV: 'development',
      },
    },
  ],
};
