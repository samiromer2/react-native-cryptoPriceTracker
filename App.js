import React, { useRef, useMemo, useState, useEffect } from "react";
import { StyleSheet, Text, View, RefreshControl } from "react-native";
import { FlatList } from "react-native-gesture-handler";
import { SafeAreaView } from "react-native-safe-area-context";
import Listitem from "./components/Listitem";
import Chart from "./components/Chart";
import { SAMPLE_DATA } from "./assets/data/sampleData";
import {getMarketData} from './services/cryptoService';
import {
  BottomSheetModal,
  BottomSheetModalProvider,
  BottomSheetBackdrop,
} from "@gorhom/bottom-sheet";
import { GestureHandlerRootView } from "react-native-gesture-handler";
const ListHeader = () => (
  <>
    <View style={styles.titleWrapper}>
      <Text style={styles.bigTitle}>Markets</Text>
    </View>
    <View style={styles.divider} />
  </>
);
export default function App() {
  const [data,setData] = useState([]);
  const [selectedCoinData, setSelectedCoinData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);

  const fetchMarketData = async (isRefresh = false) => {
    try {
      if (isRefresh) {
        setRefreshing(true);
      } else {
        setLoading(true);
      }
      setError(null);
      const marketData = await getMarketData();
      if (marketData && marketData.length > 0) {
        setData(marketData);
      } else {
        // Fallback to sample data if API fails
        setData(SAMPLE_DATA);
      }
    } catch (error) {
      setError(error.message || "Failed to fetch market data");
      // Fallback to sample data
      setData(SAMPLE_DATA);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(()=> {
    fetchMarketData();
  },[])

  const onRefresh = React.useCallback(() => {
    fetchMarketData(true);
  }, []);

  const bottomSheetModalRef = useRef(null);
  const snapPoints = useMemo(() => ["50%", "75%"], []);
  
  const openModal = React.useCallback((item) => {
    console.log("Opening modal for:", item?.name);
    if (!item) {
      console.log("Item is missing");
      return;
    }
    
    console.log("Modal ref:", bottomSheetModalRef.current);
    setSelectedCoinData(item);
    
    // Wait for state to update, then present
    setTimeout(() => {
      if (bottomSheetModalRef.current) {
        console.log("Calling present()");
        try {
          const result = bottomSheetModalRef.current.present();
          console.log("Present result:", result);
        } catch (error) {
          console.error("Error presenting modal:", error);
          console.error("Error stack:", error.stack);
        }
      } else {
        console.error("Modal ref is null!");
      }
    }, 0);
  }, []);

  const closeModal = React.useCallback(() => {
    bottomSheetModalRef.current?.dismiss();
  }, []);

  // Render backdrop
  const renderBackdrop = React.useCallback(
    (props) => (
      <BottomSheetBackdrop
        {...props}
        disappearsOnIndex={-1}
        appearsOnIndex={0}
        opacity={0.5}
      />
    ),
    []
  );
  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <BottomSheetModalProvider>
        <SafeAreaView style={styles.container}>
        {loading ? (
          <View style={styles.loadingContainer}>
            <Text style={styles.loadingText}>Loading markets...</Text>
          </View>
        ) : error && data.length === 0 ? (
          <View style={styles.errorContainer}>
            <Text style={styles.errorText}>Error: {error}</Text>
            <Text style={styles.errorSubtext}>Using sample data</Text>
          </View>
        ) : data.length === 0 ? (
          <View style={styles.emptyContainer}>
            <Text style={styles.emptyText}>No data available</Text>
          </View>
        ) : (
          <FlatList
            keyExtractor={(item) => item.id}
            data={data}
            renderItem={({ item }) => (
              <Listitem
                name={item.name}
                symbol={item.symbol}
                currentPrice={item.current_price}
                priceChangePercentage7d={
                  item.price_change_percentage_7d_in_currency || 0
                }
                logoUrl={item.image}
                onPress={() => openModal(item)}
              />
            )}
            ListHeaderComponent={<ListHeader />}
            refreshControl={
              <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
            }
          />
        )}
      </SafeAreaView>
      <BottomSheetModal
        ref={bottomSheetModalRef}
        index={0}
        snapPoints={snapPoints}
        enablePanDownToClose={true}
        enableDismissOnClose={true}
        backgroundStyle={styles.bottomSheetBackground}
        handleIndicatorStyle={styles.handleIndicator}
        backdropComponent={renderBackdrop}
        enableDynamicSizing={false}
        onChange={(index) => {
          console.log("Bottom sheet index changed:", index);
          if (index === -1) {
            console.log("Modal closed");
          } else {
            console.log("Modal opened at index:", index);
          }
        }}
        onDismiss={() => {
          console.log("Bottom sheet dismissed");
          setSelectedCoinData(null);
        }}
        onAnimate={(fromIndex, toIndex) => {
          console.log("Animating from", fromIndex, "to", toIndex);
        }}
      >
        <View style={styles.bottomSheetContent}>
          {selectedCoinData ? (
            <Chart
              currentPrice={selectedCoinData.current_price}
              logoUrl={selectedCoinData.image}
              name={selectedCoinData.name}
              symbol={selectedCoinData.symbol}
              priceChangePercentage7d={
                selectedCoinData.price_change_percentage_7d_in_currency || 0
              }
              sparkline={selectedCoinData.sparkline_in_7d?.price || []}
            />
          ) : (
            <View style={styles.loadingContainer}>
              <Text>No data selected</Text>
            </View>
          )}
        </View>
      </BottomSheetModal>
      </BottomSheetModalProvider>
    </GestureHandlerRootView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F5F7FA",
  },
  titleWrapper: {
    marginTop: 20,
    paddingHorizontal: 20,
    marginBottom: 8,
  },
  bigTitle: {
    fontSize: 34,
    fontWeight: "700",
    color: "#1A1F36",
    letterSpacing: -0.5,
  },
  divider: {
    height: 1,
    backgroundColor: "#E8ECF0",
    marginHorizontal: 20,
    marginTop: 20,
  },
  bottomSheet: {
    shadowColor: "#000000",
    shadowOffset: {
      width: 0,
      height: -4,
    },
    shadowOpacity: 0.58,
    shadowRadius: 16.0,
    elevation: 24,
  },
  bottomSheetBackground: {
    backgroundColor: "#FFFFFF",
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    shadowColor: "#000",
    shadowOffset: {
      width: 0,
      height: -2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 8,
  },
  handleIndicator: {
    backgroundColor: "#D1D5DB",
    width: 48,
    height: 5,
    borderRadius: 3,
  },
  bottomSheetContent: {
    flex: 1,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  loadingText: {
    fontSize: 16,
    color: "#6B7280",
    fontWeight: "500",
  },
  errorContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
  },
  errorText: {
    fontSize: 18,
    color: "#EF4444",
    textAlign: "center",
    marginBottom: 8,
    fontWeight: "600",
  },
  errorSubtext: {
    fontSize: 14,
    color: "#6B7280",
    textAlign: "center",
  },
  emptyContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  emptyText: {
    fontSize: 16,
    color: "#6B7280",
    fontWeight: "500",
  },
});
